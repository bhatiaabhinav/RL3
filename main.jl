using Revise

include("gw.jl")
include("learned_mdp.jl")
include("hooks.jl")
include("VAMDP.jl")
include("main_helper_fns.jl")

problem_map = Dict(
    "mdps" => make_random_mdps,
    "bandits" => make_bandits,
    "gridworlds" => make_gridworlds
)

task_horizon_map = Dict(
    "mdps" => 10,
    "bandits" => 1,
    "gridworlds" => 10000000
)

using ArgParse

args_setting = ArgParseSettings()

@add_arg_table! args_setting begin
    "problem"
        help = "problem name: mdps, bandits, gridworlds"
        required = true
    "algo"
        help = "either rl2, rl3 or rl3_coarse"
        required = true
    "iters"
        help = "how many episodes"
        arg_type = Int
        required = true
    "--suffix"
        help = "experiment_name_suffix"
        default = ""
    "--horizon", "-H"
        help = "metamdp horizon"
        arg_type = Int
        default = 100
    "--nstates", "-m"
        help = "number of state for random mdps"
        arg_type = Int
        default = 10
    "--nactions", "-n"
        help = "number of actions for random mdps"
        arg_type = Int
        default = 5
    "--lr"
        help = "learning rate"
        arg_type = Float64
        default = 0.0003
    "--decay_lr"
        help = "whether to anneal learning rate"
        action = :store_true
    "--batch_size", "-B"
        help = "batch_size"
        arg_type = Int
        default = 32000
    "--nsteps", "-M"
        help = "number of steps in each iteration of PPO. When set to 0 (default), it is set to be equal to the metamdp horizon (H)"
        arg_type = Int
        default = 0
    "--minibatch_size", "-b"
        help = "minibatch_size"
        arg_type = Int
        default = 3200
    "--model"
        help = "model to use: either markov, rnn or transformer"
        default = "transformer"
    "--dmodel"
        help = "size of the model"
        arg_type = Int
        default = 64
    "--seed"
        help = "global random seed"
        arg_type = Int
        default = 0
    "--device"
        help = "which device? cpu or gpu"
        default = "gpu"
    "--log_interval"
        help = "print states after these many episodes"
        arg_type = Int
        default = 1
    "--advantagenorm"
        help = "normalise advantage"
        arg_type = Bool
        default = true
    "--adam_eps"
        help = "epsilon in Adam Optimiser"
        arg_type = Float64
        default = 1e-7
    "--adam_wd"
        help = "Adam weight decay"
        arg_type = Float64
        default = 0.01
    "--clipnorm"
        help = "clip gradients by global norm"
        arg_type = Float32
        default = Inf32
    "--nepochs", "-K"
        help = "max number of actor-critic updates per training batch in PPO"
        arg_type = Int
        default = 8
    "--ent_bonus", "-e"
        help = "for ppo"
        arg_type = Float64
        default = 0.01
    "--decay_ent_bonus"
        help = "whether to anneal entropy bonus"
        action = :store_true
    "--kl_target"
        help = "maximum KL div between old and new policy for PPO"
        arg_type = Float64
        default = 0.01
    "--ppo_epsilon"
        help = "for PPO clip objective"
        arg_type = Float64
        default = 0.2
    "--lambda", "-l"
        help = "for generalized advantage estimation"
        arg_type = Float64
        default = 0.3
    "--progressmeter"
        help = "whether to request algorithms to show internal progress using a progressbar"
        action = :store_true
    "--video"
        help = "whether to record video for gridworlds"
        action = :store_true
    "--video_interval"
        help = "Video record interval for grridworlds"
        arg_type = Int
        default = 5
    "--act_greedy", "-g"
        help = "Whether to act greedy when interacting with main env to generate scores, while PPO learning happens in the background"
        action = :store_true
    "--nheads"
        help = "number of heads in transformer"
        arg_type = Int
        default = 4
    "--ndecoders"
        help = "number of decoder layers in transformer"
        arg_type = Int
        default = 2
    "--test_model"
        help = "model path to test"
        arg_type = String
        default = ""
    "--extrapolate_mode"
        help = "whether to extrapolate a model trained on half horizon to full horizon"
        action = :store_true
    "--ood"
        help = "whether to test on ood data"
        action = :store_true
    "--grid_variation"
        help = "either of 11x11, 13x13, 13x13_dense, 13x13_deterministic, 13x13_watery, 13x13_dangerous, 13x13_corner"
        arg_type = String
        default = "11x11"
    "--model_save_interval"
        help = "how often to save model"
        arg_type = Int
        default = 100
end


function run_experiments(_args)
    kwargs = parse_args(_args, args_setting; as_symbols=true)
    kwargs[:narms] = kwargs[:nactions]
    if kwargs[:nsteps] == 0
        kwargs[:nsteps] = kwargs[:horizon]
    end
    kwargs[:nenvs] = kwargs[:batch_size] ÷ kwargs[:nsteps]
    kwargs[:task_horizon] = min(task_horizon_map[kwargs[:problem]], kwargs[:horizon])
    if kwargs[:extrapolate_mode]
        kwargs[:extrapolate_from] = kwargs[:horizon] ÷ 2
    else
        kwargs[:extrapolate_from] = kwargs[:horizon]
    end
    if kwargs[:algo] == "rl3_coarse"
        kwargs[:abstraction_cluster_size] = 2
        kwargs[:abstraction_radius] = contains(kwargs[:grid_variation], "13x13") ? 0.08 : 0.1
    else
        kwargs[:abstraction_cluster_size] = 1
        kwargs[:abstraction_radius] = 0.0
    end
    if contains(kwargs[:grid_variation], "deterministic")
        global slip_probabality
        slip_probabality = slip_probabality_determinisitc
    else
        global slip_probabality
        slip_probabality = slip_probabality_stochastic
    end
    if kwargs[:suffix] != ""; kwargs[:suffix] = "-" * kwargs[:suffix]; end
    experiment_name = "$(kwargs[:problem])-$(kwargs[:algo])$(kwargs[:suffix])-$(kwargs[:seed])"
    @info "Experiment configuration" experiment_name kwargs...
    kwargs[:config] = deepcopy(kwargs)
    kwargs[:device] = eval(Symbol(kwargs[:device]))

    seed = kwargs[:seed]
    Random.seed!(seed)

    function generate_problem_set(set_seed; kwargs...)
        set_kwargs = Dict(kwargs...)
        set_kwargs[:seed] = set_seed
        _problem = problem_map[kwargs[:problem]](; set_kwargs...)
        set_seed == seed && println("random policy score on this metamdp = ", test_random_policy(_problem; kwargs...)[1])
        if kwargs[:algo] == "rl3" || kwargs[:algo] == "rl3_coarse"
            set_seed == seed && println("making value augmented mdps")
            Q_DENOM = kwargs[:problem] == "bandits" ? 1f0 : 100f0
            VI_EP = kwargs[:problem] == "gridworlds" ? 0.1 : 0.01
            _problem = wrap_VAMDPs(_problem; task_horizon=kwargs[:task_horizon], abstraction_radius=kwargs[:abstraction_radius], abstraction_cluster_size=kwargs[:abstraction_cluster_size], Q_DENOM=Q_DENOM, VI_EP=VI_EP, drop_observation=(kwargs[:problem] == "bandits"))
        else
            if kwargs[:problem] ∈ ["bandits", "mdps"]
                _problem = wrap_onehot_mdps(_problem)
            end
        end
        return _problem
    end

    println("creating problem mdps")
    problem = generate_problem_set(seed; kwargs...)
    problem_batch = [generate_problem_set(seed + 1000 + i; kwargs...) for i in 1:kwargs[:nenvs]] # for parallel envs for ppo
    kwargs[:problem_batch] = problem_batch
    
    iters = kwargs[:iters]
    println("Running experiment ", experiment_name)
    do_ppo_learning(experiment_name, problem, iters; problem_batch=problem_batch, kwargs...)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_experiments(ARGS);
end
