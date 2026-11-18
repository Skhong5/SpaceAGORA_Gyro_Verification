
# ==============================================================
# GYROSCOPE CONSTANT BIAS SWEEP
#
# Sensor verification with a shared true-rate trajectory.
#
# Gyro 1: Ideal gyro used for spacecraft control
# Gyro 2: Biased gyro used for measurement testing
#
# All bias cases have the same spacecraft dynamics.
# ==============================================================

module GyroBiasReferenceSweep

using SpaceAGORA
using StaticArrays
using Statistics
using Printf
using Plots

# ==============================================================
# 1. EXPERIMENT SETTINGS
# ==============================================================

const BIAS_VALUES = [-0.003, -0.001, 0.0, 0.001, 0.003]

const ANGLE_RANDOM_WALK = 0.0
const RATE_RANDOM_WALK = 0.0

const GYRO_PERIOD_S = 1.0
const MISSION_S = 120.0
const SEED = 42

# ==============================================================
# 2. SHARED ESTIMATE
# ==============================================================

mutable struct RateEstimate
    ω_body::SVector{3, Float64}
    time_s::Float64
end

RateEstimate() = RateEstimate(
    zero(SVector{3, Float64}), NaN
)

SpaceAGORA.link_cardinality(::RateEstimate) =
    (producers=1:1, consumers=0:typemax(Int))

function SpaceAGORA.link_initialize!(
    e::RateEstimate, producers, consumers, u, p, t
)
    e.ω_body = zero(SVector{3, Float64})
    e.time_s = NaN
    return nothing
end

# ==============================================================
# 3. GYROSCOPE ESTIMATOR
# ==============================================================

struct GyroRateEstimator <: SpaceAGORA.AbstractNavigationModel
    spacecraft_index::Int
    gyro::Int
    estimate::RateEstimate
end

SpaceAGORA.spacecraft_scope(n::GyroRateEstimator) =
    BoundSpacecraft((n.spacecraft_index,))

SpaceAGORA.gnc_links(n::GyroRateEstimator) =
    (Produces(n.estimate),)

function SpaceAGORA.update_navigation!(
    n::GyroRateEstimator, u, p, t, spacecraft_index
)
    spacecraft =
        p.args.dynamics_model.spacecraft[spacecraft_index]

    n.estimate.ω_body = SVector{3, Float64}(
        sensor_measurement(spacecraft, n.gyro)
    )

    n.estimate.time_s =
        sensor_measurement_time(spacecraft, n.gyro)

    return nothing
end

SpaceAGORA.checkpoint_state(n::GyroRateEstimator) =
    (
        ω_body=n.estimate.ω_body,
        time_s=n.estimate.time_s
    )

function SpaceAGORA.restore_checkpoint_state!(
    n::GyroRateEstimator, data
)
    n.estimate.ω_body = data.ω_body
    n.estimate.time_s = data.time_s
    return nothing
end

# ==============================================================
# 4. RATE-DAMPING CONTROLLER
# ==============================================================

Base.@kwdef mutable struct EstimatedRateDamper <:
    SpaceAGORA.AbstractControlEffectorModel

    spacecraft_index::Int = 1
    gain_n_m_s::Float64 = 2.0
    max_torque_n_m::Float64 = 1.0
    period_s::Float64 = 1.0
    estimate::RateEstimate

    torque_body::SVector{3, Float64} =
        zero(SVector{3, Float64})

    tick_s::Float64 = NaN
end

SpaceAGORA.spacecraft_scope(c::EstimatedRateDamper) =
    BoundSpacecraft((c.spacecraft_index,))

SpaceAGORA.gnc_period_s(c::EstimatedRateDamper) =
    c.period_s

SpaceAGORA.gnc_links(c::EstimatedRateDamper) =
    (Consumes(c.estimate),)

SpaceAGORA.held_command(
    c::EstimatedRateDamper, spacecraft_index
) = c.torque_body

SpaceAGORA.held_command_time(
    c::EstimatedRateDamper, spacecraft_index
) = c.tick_s

function SpaceAGORA.release_command!(
    c::EstimatedRateDamper, t, spacecraft_index
)
    c.torque_body = zero(SVector{3, Float64})
    c.tick_s = t
    return nothing
end

function SpaceAGORA.initialize_control!(
    c::EstimatedRateDamper, u, p, t
)
    c.torque_body = zero(SVector{3, Float64})
    c.tick_s = NaN
    return nothing
end

function SpaceAGORA.update_control!(
    c::EstimatedRateDamper, u, p, t, spacecraft_index
)
    c.torque_body = clamp.(
        -c.gain_n_m_s .* c.estimate.ω_body,
        -c.max_torque_n_m,
        c.max_torque_n_m
    )

    c.tick_s = t
    return nothing
end

function SpaceAGORA.control_wrench(
    c::EstimatedRateDamper, x::StateSample, p, t
)
    zero3 = zero(SVector{3, Float64})

    x.spacecraft_index == c.spacecraft_index ||
        return zero3, zero3

    return zero3, c.torque_body
end

SpaceAGORA.checkpoint_state(c::EstimatedRateDamper) =
    (
        torque_body=c.torque_body,
        tick_s=c.tick_s
    )

function SpaceAGORA.restore_checkpoint_state!(
    c::EstimatedRateDamper, data
)
    c.torque_body = data.torque_body
    c.tick_s = data.tick_s
    return nothing
end

# ==============================================================
# 5. CREATE SIMULATION CONFIGURATION
# ==============================================================

function study_configuration(bias_z)

    root = Link(
        root=true,
        m=500.0,
        dims=MVector(1.0, 1.0, 1.0)
    )

    ω0 = 0.05

    ic = CartesianInitialCondition(
        SVector(4.0e9, 0.0, 0.0),
        SVector(0.0, 0.0, 0.0);
        q=SVector(0.0, 0.0, 0.0, 1.0),
        ang_vel=SVector(0.0, 0.0, ω0)
    )

    spacecraft = SpacecraftModel(
        links=[root],
        root=root,
        prop_mass=0.0,
        inertia_tensor=root.inertia,
        initial_condition=ic,
        id=1
    )

    # ----------------------------------------------------------
    # GYRO 1: IDEAL GYRO USED BY THE CONTROLLER
    #
    # Sensor index 1
    # Output columns: sc1_sensor_1, 2, 3
    # ----------------------------------------------------------

    add_sensor!(
        spacecraft,
        root,
        RateGyro(
            period_s=GYRO_PERIOD_S,
            angle_random_walk=0.0,
            rate_random_walk=0.0,
            initial_bias=SVector(0.0, 0.0, 0.0)
        )
    )

    # ----------------------------------------------------------
    # GYRO 2: EXPERIMENTAL GYRO WITH ADJUSTABLE BIAS
    #
    # Sensor index 2
    # Output columns: sc1_sensor_4, 5, 6
    # ----------------------------------------------------------

    add_sensor!(
        spacecraft,
        root,
        RateGyro(
            period_s=GYRO_PERIOD_S,
            angle_random_walk=ANGLE_RANDOM_WALK,
            rate_random_walk=RATE_RANDOM_WALK,
            initial_bias=SVector(0.0, 0.0, bias_z),
            seed=SEED
        )
    )

    estimate = RateEstimate()

    # IMPORTANT:
    # The estimator reads sensor 1, the ideal gyro.
    # Therefore the experimental bias does not
    # influence spacecraft control.

    estimator = GyroRateEstimator(1, 1, estimate)

    damper = EstimatedRateDamper(
        estimate=estimate
    )

    planet = make_no_gram_planet(:earth)

    return SimulationConfiguration(

        simulation_settings=SimulationSettings(
            results=false,
            verbose=false,
            generate_plots=false,
            normalize=false
        ),

        mission_configuration=MissionConfiguration(
            mission_type=MissionTime,
            mission_time=MISSION_S,
            number_of_orbits=1,
            keplerian=false,
            orientation_sim=true,
            num_steps_to_save=120,
            data_rate=GYRO_PERIOD_S
        ),

        environment_model=EnvironmentModel(
            planet=planet,
            EI=120.0,
            density_model=NoAtmosphereModel(),
            ephemerides_model=SimpleEphemeridesModel(),
            thermal_model=MaxwellianHeat(
                thermal_accomodation_factor=1.0,
                planet=planet
            ),
            topography=false,
            wind=false
        ),

        dynamics_model=DynamicsModel([spacecraft], ()),

        guidance_model=GuidanceModel(
            guidance_effectors=(),
            guidance_rates=Float64[]
        ),

        navigation_model=NavigationModel(
            navigation_effectors=(estimator,),
            navigation_rates=[1.0]
        ),

        control_model=ControlModel(
            control_effectors=(damper,),
            control_rates=[damper.period_s]
        ),

        initial_time=InitialTime(
            year=2024,
            month=1,
            day=1,
            hour=0,
            minute=0,
            second=0.0
        ),

        integration_tolerances=IntegrationTolerances()
    )
end

# ==============================================================
# 6. RUN ONE BIAS CASE
# ==============================================================

function run_case(bias_z, outdir)

    config = study_configuration(bias_z)

    result = run_simulation(
        config;
        return_results=true,
        return_solution=true
    )

    # Simulation output timestamps
    t = Float64.(result.table.time)

    # Experimental gyro z-axis measurement
    # Sensor 2, component 3 = column 6
    measured = Float64.(
        result.table.sc1_sensor_6
    )

    # True spacecraft z-axis angular velocity
    truth = [
        result.solution(ti).sc[1].ω[3]
        for ti in t
    ]

    # Measurement error
    error = measured .- truth

    # Statistics
    mean_error = mean(error)
    std_error = std(error)

    max_mismatch = maximum(
        abs.(error .- bias_z)
    )

    # Save results for this bias
    tag = replace(
        @sprintf("%+.6f", bias_z),
        "+"=>"plus_",
        "-"=>"minus_",
        "."=>"p"
    )

    csv_path = joinpath(
        outdir, "bias_$(tag).csv"
    )

    open(csv_path, "w") do io

        println(
            io,
            "time_s,true_z_rad_s,measured_z_rad_s,error_z_rad_s"
        )

        for i in eachindex(t)
            @printf(
                io,
                "%.9f,%.12g,%.12g,%.12g\n",
                t[i], truth[i], measured[i], error[i]
            )
        end
    end

    return (
        bias=bias_z,
        time=t,
        truth=truth,
        measured=measured,
        error=error,
        mean_error=mean_error,
        std_error=std_error,
        max_mismatch=max_mismatch
    )
end

# ==============================================================
# 7. RUN BIAS SWEEP
# ==============================================================

function run_sweep()

    outdir = joinpath(
        @__DIR__, "gyro_bias_sweep_outputs"
    )

    mkpath(outdir)

    cases = []

    println("\n========= GYROSCOPE BIAS SWEEP =========")

    println(
        "Configured Bias    Mean Error     Std Error    Max Mismatch"
    )

    # Run each bias value
    for bias_z in BIAS_VALUES

        case = run_case(bias_z, outdir)
        push!(cases, case)

        @printf(
            "%+14.6f  %+12.7f   %.3e    %.3e\n",
            case.bias,
            case.mean_error,
            case.std_error,
            case.max_mismatch
        )
    end

    # All cases should have the same true trajectory
    reference = cases[1]

    for c in cases
        @assert length(c.time) == length(reference.time)

        time_difference = maximum(
            abs.(c.time .- reference.time)
        )

        truth_difference = maximum(
            abs.(c.truth .- reference.truth)
        )

        @printf(
            "Bias %+.4f: max true-rate difference = %.3e rad/s\n",
            c.bias, truth_difference
        )

        @assert time_difference < 1e-8
        @assert truth_difference < 1e-7
    end

    # ----------------------------------------------------------
    # PLOT 1: TRUE RATE AND ALL GYRO MEASUREMENTS
    # ----------------------------------------------------------

    p1 = plot(
        reference.time,
        reference.truth;
        label="True angular rate",
        color=:black,
        linestyle=:dash,
        linewidth=3,
        title="True and measured gyroscope z-axis rates",
        xlabel="Time (s)",
        ylabel="Angular rate (rad/s)",
        legend=:topright
    )

    for c in cases

        plot!(
            p1,
            c.time,
            c.measured;
            label=@sprintf("Bias = %+.4f rad/s", c.bias),
            linewidth=2
        )
    end

    # ----------------------------------------------------------
    # PLOT 2: MEASUREMENT ERROR FOR EACH BIAS
    # ----------------------------------------------------------

    p2 = plot(
        title="Gyroscope measurement error",
        xlabel="Time (s)",
        ylabel="Measured - true (rad/s)",
        legend=:topright
    )

    for c in cases

        plot!(
            p2,
            c.time,
            c.error;
            label=@sprintf("Bias = %+.4f rad/s", c.bias),
            linewidth=2
        )
    end

    # ----------------------------------------------------------
    # COMBINE AND SAVE
    # ----------------------------------------------------------

    fig = plot(
        p1,
        p2;
        layout=(2, 1),
        size=(1100, 800)
    )

    display(fig)

    savefig(
        fig,
        joinpath(outdir, "gyro_bias_sweep.png")
    )

    # ----------------------------------------------------------
    # SAVE SUMMARY CSV
    # ----------------------------------------------------------

    summary_path = joinpath(
        outdir, "bias_sweep_summary.csv"
    )

    open(summary_path, "w") do io

        println(
            io,
            "bias_rad_s,mean_error_rad_s," *
            "std_error_rad_s,max_mismatch_rad_s"
        )

        for c in cases

            @printf(
                io,
                "%.12g,%.12g,%.12g,%.12g\n",
                c.bias,
                c.mean_error,
                c.std_error,
                c.max_mismatch
            )
        end
    end

    println("\nSaved results to: ", outdir)

    return (
        cases=cases,
        plot=fig,
        outdir=outdir
    )
end

end # module

# ==============================================================
# 8. EXECUTE
# ==============================================================

bias_sweep = GyroBiasReferenceSweep.run_sweep()

nothing
