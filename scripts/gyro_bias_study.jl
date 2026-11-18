# SpaceAGORA gyroscope bias study
# Based on examples/gnc_algorithm_env/estimator.jl from commit
# dafb5f10de8c7497efe3e898e9e2630a458b7a7c
#
# Put this file in examples/gnc_algorithm_env/ of your working checkout.
# Run from the repository root using the gnc_algorithm_env project.
# Edit BIAS_Z below and run the file again for one bias case at a time.
# This is an experimental study: the original exact-gyro hand-calculation
# assertion and the conformance tests are intentionally omitted.

module GyroBiasStudy

using SpaceAGORA
using StaticArrays
using Statistics
using Printf
using Plots

# ==============================================================
# CHANGE THESE SETTINGS FOR EACH EXPERIMENT
# ==============================================================
const BIAS_Z = 0.0005          # z-axis initial gyro bias (rad/s)
const ANGLE_RANDOM_WALK = 0.0 # gyro white-noise intensity (rad/sqrt(s))
const RATE_RANDOM_WALK = 0.0  # bias random-walk intensity (rad/s/sqrt(s))
const GYRO_PERIOD_S = 1.0     # gyro sample period (s)
const MISSION_S = 120.0       # simulation duration (s)
const SEED = 42               # reproducible random-number stream

# ==============================================================
# 1. SHARED NAVIGATION ESTIMATE (copied from estimator.jl)
# ==============================================================
mutable struct RateEstimate
    ω_body::SVector{3, Float64}
    time_s::Float64
end
RateEstimate() = RateEstimate(zero(SVector{3, Float64}), NaN)

SpaceAGORA.link_cardinality(::RateEstimate) =
    (producers=1:1, consumers=0:typemax(Int))

function SpaceAGORA.link_initialize!(e::RateEstimate, producers, consumers, u, p, t)
    e.ω_body, e.time_s = zero(SVector{3, Float64}), NaN
    return nothing
end

# ==============================================================
# 2. ESTIMATOR (copies the most recent gyro reading)
# ==============================================================
struct GyroRateEstimator <: SpaceAGORA.AbstractNavigationModel
    spacecraft_index::Int
    gyro::Int
    estimate::RateEstimate
end

SpaceAGORA.spacecraft_scope(n::GyroRateEstimator) = BoundSpacecraft((n.spacecraft_index,))
SpaceAGORA.gnc_links(n::GyroRateEstimator) = (Produces(n.estimate),)

function SpaceAGORA.update_navigation!(n::GyroRateEstimator, u, p, t, spacecraft_index)
    spacecraft = p.args.dynamics_model.spacecraft[spacecraft_index]
    n.estimate.ω_body = SVector{3, Float64}(sensor_measurement(spacecraft, n.gyro))
    n.estimate.time_s = sensor_measurement_time(spacecraft, n.gyro)
    return nothing
end

SpaceAGORA.checkpoint_state(n::GyroRateEstimator) =
    (ω_body=n.estimate.ω_body, time_s=n.estimate.time_s)

function SpaceAGORA.restore_checkpoint_state!(n::GyroRateEstimator, data)
    n.estimate.ω_body, n.estimate.time_s = data.ω_body, data.time_s
    return nothing
end

# ==============================================================
# 3. RATE-DAMPING CONTROLLER (same as estimator.jl)
# ==============================================================
Base.@kwdef mutable struct EstimatedRateDamper <: SpaceAGORA.AbstractControlEffectorModel
    spacecraft_index::Int = 1
    gain_n_m_s::Float64 = 2.0
    max_torque_n_m::Float64 = 1.0
    period_s::Float64 = 1.0
    estimate::RateEstimate
    torque_body::SVector{3, Float64} = zero(SVector{3, Float64})
    tick_s::Float64 = NaN
end

SpaceAGORA.spacecraft_scope(c::EstimatedRateDamper) = BoundSpacecraft((c.spacecraft_index,))
SpaceAGORA.gnc_period_s(c::EstimatedRateDamper) = c.period_s
SpaceAGORA.gnc_links(c::EstimatedRateDamper) = (Consumes(c.estimate),)
SpaceAGORA.held_command(c::EstimatedRateDamper, spacecraft_index) = c.torque_body
SpaceAGORA.held_command_time(c::EstimatedRateDamper, spacecraft_index) = c.tick_s

function SpaceAGORA.release_command!(c::EstimatedRateDamper, t, spacecraft_index)
    c.torque_body, c.tick_s = zero(SVector{3, Float64}), t
    return nothing
end

function SpaceAGORA.initialize_control!(c::EstimatedRateDamper, u, p, t)
    c.torque_body, c.tick_s = zero(SVector{3, Float64}), NaN
    return nothing
end

function SpaceAGORA.update_control!(c::EstimatedRateDamper, u, p, t, spacecraft_index)
    c.torque_body = clamp.(-c.gain_n_m_s .* c.estimate.ω_body,
                           -c.max_torque_n_m, c.max_torque_n_m)
    c.tick_s = t
    return nothing
end

function SpaceAGORA.control_wrench(c::EstimatedRateDamper, x::StateSample, p, t)
    zero3 = zero(SVector{3, Float64})
    x.spacecraft_index == c.spacecraft_index || return zero3, zero3
    return zero3, c.torque_body
end

SpaceAGORA.checkpoint_state(c::EstimatedRateDamper) =
    (torque_body=c.torque_body, tick_s=c.tick_s)

function SpaceAGORA.restore_checkpoint_state!(c::EstimatedRateDamper, data)
    c.torque_body, c.tick_s = data.torque_body, data.tick_s
    return nothing
end

# ==============================================================
# 4. BUILD A FRESH SPACECRAFT FOR THIS ONE BIAS CASE
# ==============================================================
function study_configuration()
    root = Link(root=true, m=500.0, dims=MVector(1.0, 1.0, 1.0))
    ω0 = 0.05
    ic = CartesianInitialCondition(
        SVector(4.0e9, 0.0, 0.0), SVector(0.0, 0.0, 0.0);
        q=SVector(0.0, 0.0, 0.0, 1.0),
        ang_vel=SVector(0.0, 0.0, ω0)
    )
    spacecraft = SpacecraftModel(
        links=[root], root=root, prop_mass=0.0,
        inertia_tensor=root.inertia, initial_condition=ic, id=1
    )

    # Bias on z only; x/y are unbiased. No noise unless enabled above.
    add_sensor!(spacecraft, root, RateGyro(
        period_s=GYRO_PERIOD_S,
        initial_bias=SVector(0.0, 0.0, BIAS_Z),
        angle_random_walk=ANGLE_RANDOM_WALK,
        rate_random_walk=RATE_RANDOM_WALK,
        seed=SEED
    ))

    estimate = RateEstimate()
    estimator = GyroRateEstimator(1, 1, estimate)
    damper = EstimatedRateDamper(estimate=estimate)
    planet = make_no_gram_planet(:earth)

    return SimulationConfiguration(
        simulation_settings=SimulationSettings(
            results=false, verbose=false, generate_plots=false, normalize=false
        ),
        mission_configuration=MissionConfiguration(
            mission_type=MissionTime, mission_time=MISSION_S, number_of_orbits=1,
            keplerian=false, orientation_sim=true, num_steps_to_save=120,
            data_rate=GYRO_PERIOD_S
        ),
        environment_model=EnvironmentModel(
            planet=planet, EI=120.0, density_model=NoAtmosphereModel(),
            ephemerides_model=SimpleEphemeridesModel(),
            thermal_model=MaxwellianHeat(
                thermal_accomodation_factor=1.0, planet=planet
            ),
            topography=false, wind=false
        ),
        dynamics_model=DynamicsModel([spacecraft], ()),
        guidance_model=GuidanceModel(guidance_effectors=(), guidance_rates=Float64[]),
        navigation_model=NavigationModel(
            navigation_effectors=(estimator,), navigation_rates=[1.0]
        ),
        control_model=ControlModel(
            control_effectors=(damper,), control_rates=[damper.period_s]
        ),
        initial_time=InitialTime(
            year=2024, month=1, day=1, hour=0, minute=0, second=0.0
        ),
        integration_tolerances=IntegrationTolerances()
    )
end

# ==============================================================
# 5. RUN, COMPARE MEASURED VS TRUE BODY RATE, PLOT AND EXPORT
# ==============================================================
function run_study()
    config = study_configuration()

    # SpaceAGORA returns a results table AND a numerical ODE solution.
    result = run_simulation(config; return_results=true, return_solution=true)
    df = result.table

    # Results table columns for the 1st 3-axis sensor on spacecraft 1:
    # sc1_sensor_1 = x, sc1_sensor_2 = y, sc1_sensor_3 = z (rad/s).
    times = Float64.(df.time)
    measured_z = Float64.(df.sc1_sensor_3)

    # The default table does not include true body angular rate.
    # The ODE solution can be evaluated at the same times instead.
    true_z = [result.solution(t).sc[1].ω[3] for t in times]
    error_z = measured_z .- true_z

    # Statistics are for error (measurement - true rate), not raw gyro data:
    # raw gyro data includes changing spacecraft motion due to the controller.
    mean_error = mean(error_z)
    std_error = std(error_z)
    expected_white_std = ANGLE_RANDOM_WALK / sqrt(GYRO_PERIOD_S)

    println("\n========== GYRO BIAS STUDY ==========")
    @printf("Z-axis initial bias:        %+.8f rad/s\n", BIAS_Z)
    @printf("Sampling period:            %.3f s\n", GYRO_PERIOD_S)
    @printf("Angle random walk (ARW):    %.6g rad/sqrt(s)\n", ANGLE_RANDOM_WALK)
    @printf("Rate random walk (RRW):     %.6g rad/s/sqrt(s)\n", RATE_RANDOM_WALK)
    @printf("Mean measurement error:     %+.8f rad/s\n", mean_error)
    @printf("Std dev of error:           %.8g rad/s\n", std_error)
    @printf("Expected white-noise std:   %.8g rad/s\n", expected_white_std)
    @printf("Final true z rate:          %+.8f rad/s\n", true_z[end])
    @printf("Final measured z rate:      %+.8f rad/s\n", measured_z[end])
    println("Number of output samples:   ", length(times))

    if ANGLE_RANDOM_WALK == 0 && RATE_RANDOM_WALK == 0
        @printf("Max |error - set bias|:     %.4g rad/s\n", maximum(abs.(error_z .- BIAS_Z)))
    else
        println("Note: std(error) matches ARW/sqrt(period) only statistically")
        println("when rate_random_walk=0 and the sample grid matches gyro ticks.")
    end

    println("\nFirst five saved samples:")
    println(" time (s) | true z (rad/s) | measured z (rad/s) | error (rad/s)")
    for i in 1:min(5, length(times))
        @printf(" %8.2f | %14.8f | %18.8f | %+14.8f\n",
                times[i], true_z[i], measured_z[i], error_z[i])
    end

    p1 = plot(times, true_z;
              label="True z rate", linewidth=2,
              xlabel="Time (s)", ylabel="Angular rate (rad/s)",
              title="Gyroscope: true vs measured z-axis rate")
    plot!(p1, times, measured_z;
          label="Gyro measured z rate", linewidth=1.7, linestyle=:dash)

    p2 = plot(times, error_z;
              label="Measured - true", linewidth=2,
              xlabel="Time (s)", ylabel="Error (rad/s)",
              title="Gyroscope z-axis measurement error")
    hline!(p2, [BIAS_Z]; label="Specified initial bias", linestyle=:dash)

    fig = plot(p1, p2; layout=(2, 1), size=(1000, 750))
    display(fig) # Shows in VS Code's Julia plot viewer when available.

    # Save separately for each bias, so successive runs don't overwrite results.
    outdir = joinpath(@__DIR__, "gyro_study_outputs")
    mkpath(outdir)
    tag = replace(@sprintf("%+.6f", BIAS_Z), "+"=>"plus_", "-"=>"minus_", "."=>"p")
    png_path = joinpath(outdir, "gyro_bias_$(tag).png")
    csv_path = joinpath(outdir, "gyro_bias_$(tag).csv")
    savefig(fig, png_path)

    open(csv_path, "w") do io
        println(io, "time_s,true_z_rad_s,measured_z_rad_s,error_z_rad_s")
        for i in eachindex(times)
            @printf(io, "%.9f,%.12g,%.12g,%.12g\n",
                    times[i], true_z[i], measured_z[i], error_z[i])
        end
    end

    println("\nSaved plot: ", png_path)
    println("Saved data: ", csv_path)
    return (time=times, truth=true_z, measured=measured_z, error=error_z,
            mean_error=mean_error, std_error=std_error, plot=fig)
end

end # module GyroBiasStudy

# Re-running the full file in VS Code replaces this module, avoiding
# the struct-redefinition errors that occur when pasting types at top level.
GyroBiasStudy.run_study()
nothing
