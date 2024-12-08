using Revise

include("custom_gw.jl")
include("learned_mdp.jl")
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
    "problem_name"
        help = "problem name: mdps, bandits, gridworlds"
        required = true
    "algo"
        help = "either rl2, rl3, rl3_coarse"
        required = true
    "iters"
        help = "how many episodes"
        arg_type = Int
        required = true
    "--test_episodes"
        help = "how many episodes to test on"
        arg_type = Int
        default = 1000
    "--parallel_testing"
        help = "whether to test in parallel using PPO envs"
        action = :store_true
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
        default = 32768
    "--nsteps", "-M"
        help = "number of steps in each iteration of PPO. When set to 0 (default), it is set to be equal to the metamdp horizon (H)"
        arg_type = Int
        default = 0
    "--minibatch_size", "-b"
        help = "minibatch_size"
        arg_type = Int
        default = 4096
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
        help = "which device for learning? cpu or gpu"
        default = "gpu"
    "--inference_device"
        help = "which device for inference? cpu or gpu"
        default = "cpu"
    "--log_interval"
        help = "print states after these many episodes"
        arg_type = Int
        default = 1
    "--advantagenorm"
        help = "normalise advantage"
        arg_type = Bool
        default = true
    "--obsnorm"
        help = "normalise observations"
        action = :store_true
    "--rewardnorm"
        help = "normalise rewards"
        action = :store_true
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
    "--iters_per_postepisode"
        help = "how many PPO iterations to run after each episode"
        arg_type = Int
        default = 10
    "--video"
        help = "whether to record video for gridworlds"
        action = :store_true
    "--video_interval"
        help = "Video record interval for gridworlds"
        arg_type = Int
        default = 25
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
    "--linear_attention"
        help = "whether to use linear attention in transformer"
        action = :store_true
    "--test_model"
        help = "model path to test"
        arg_type = String
        default = ""
    "--continue_model"
        help = "model path to continue training"
        arg_type = String
        default = ""
    "--ood"
        help = "whether to test on ood data"
        action = :store_true
    "--variation"
        help = "either of 11x11, 13x13, 13x13_dense, 13x13_deterministic, 13x13_watery, 13x13_dangerous, 13x13_corner"
        arg_type = String
        default = "11x11"
    "--action_num_bins"
        help = "number of bins for discretizing actions in continuous action space problems. Must be provided for continuous action space problems when using rl3"
        arg_type = Union{Vector{Int}, Nothing}
        default = nothing
    "--model_save_interval"
        help = "how often to save model"
        arg_type = Int
        default = 500
    "--no_plots"
        help = "whether to disable plots"
        action = :store_true
    "--no_multithreading"
        help = "whether to disable multithreading for PPO envs"
        action = :store_true
    "--no_pe"
        help = "Whether to disable positional encoding in transformer"
        action=:store_true
    "--no_decoder"
        help = "Whether to disable decoder in transformer"
        action=:store_true
    "--include_time_context"
        help = "Method to include time context in metamdp state: none, concat, add"
        arg_type = Symbol
        default = :concat
    "--laplace_smoothing"
        help = "Laplace smoothing coefficient for model estimation in VAMDPs"
        arg_type = Float64
        default = 0.1
    "--include_standard_errors"
        help = "Whether to include standard errors in VAMDPs"
        action=:store_true
    "--no_evidence_wrapper"
        help = "Whether to disable evidence wrapper"
        action=:store_true
    "--enable_wandb"
        help = "Whether to enable wandb logging. Requires PythonCall to be installed"
        action=:store_true
end

function run_experiments(_args)
    kwargs = parse_args(_args, args_setting; as_symbols=true)
    @assert kwargs[:algo] ∈ ["rl2", "rl3", "rl3_coarse"]
    if kwargs[:enable_wandb]
        if isnothing(wandb)
            println("wandb not available. Please install PythonCall, or remove --enable_wandb flag and try again")
            return
        end
    end
    kwargs[:narms] = kwargs[:nactions]
    if kwargs[:nsteps] == 0
        kwargs[:nsteps] = kwargs[:horizon]
    end
    if kwargs[:test_model] != "" && kwargs[:parallel_testing]
        kwargs[:batch_size] = kwargs[:horizon] * kwargs[:test_episodes]
        println("Create $(kwargs[:test_episodes]) environments for parallel testing")
    end
    kwargs[:nenvs] = kwargs[:batch_size] ÷ kwargs[:nsteps]
    kwargs[:task_horizon] = min(task_horizon_map[kwargs[:problem_name]], kwargs[:horizon])
    if kwargs[:algo] == "rl3_coarse"
        kwargs[:abstraction_cluster_size] = 2
        kwargs[:abstraction_radius] = contains(kwargs[:variation], "13x13") ? 0.08 : 0.1
    else
        kwargs[:abstraction_cluster_size] = 1
        kwargs[:abstraction_radius] = 0.0
    end
    
    if kwargs[:suffix] != ""; kwargs[:suffix] = "-" * kwargs[:suffix]; end
    project_name = "$(kwargs[:problem_name])-$(kwargs[:horizon])"
    experiment_name = "$(kwargs[:algo])$(kwargs[:suffix])-$(kwargs[:seed])"
    @info "Experiment configuration" project_name experiment_name kwargs...
    kwargs[:config] = deepcopy(kwargs)
    kwargs[:device] = eval(Symbol(kwargs[:device]))
    kwargs[:inference_device] = eval(Symbol(kwargs[:inference_device]))
    kwargs[:omit_standard_errors] = !kwargs[:include_standard_errors]

    seed = kwargs[:seed]
    Random.seed!(seed)

    function generate_problem_set(set_seed; kwargs...)
        set_kwargs = Dict(kwargs...)
        set_kwargs[:seed] = set_seed
        _problem = problem_map[kwargs[:problem_name]](; set_kwargs...)
        set_seed == seed && println("random policy score on this metamdp = ", test_random_policy(_problem; kwargs...)[1])
        if kwargs[:algo] == "rl3" || kwargs[:algo] == "rl3_coarse"
            set_seed == seed && println("making value augmented mdps")
            Q_DENOM = kwargs[:problem_name] == "bandits" ? 1f0 : 10f0
            VI_EP = kwargs[:problem_name] == "gridworlds" ? 0.1 : 0.01
            _problem = wrap_VAMDPs(_problem; laplace_smoothing=kwargs[:laplace_smoothing],  omit_standard_errors=kwargs[:omit_standard_errors], task_horizon=kwargs[:task_horizon], abstraction_radius=kwargs[:abstraction_radius], abstraction_cluster_size=kwargs[:abstraction_cluster_size], action_num_bins=kwargs[:action_num_bins], Q_DENOM=Q_DENOM, VI_EP=VI_EP)
        else
            if kwargs[:problem_name] ∈ ["bandits", "mdps"]
                _problem = wrap_onehot_mdps(_problem)
            end
        end
        return _problem
    end

    println("creating problem mdps")
    problem_set = generate_problem_set(seed; kwargs...)
    problem_set_batch = ProgressMeter.@showprogress "Creating $(kwargs[:nenvs]) parallel problem set batches for PPO" [generate_problem_set(seed + i; kwargs...) for i in 1:kwargs[:nenvs]] # for parallel envs for ppo
    kwargs[:problem_set_batch] = problem_set_batch

    iters = kwargs[:iters]
    println("Running experiment: Project=", project_name, " Experiment=", experiment_name, " Iters=", iters)
    do_ppo_learning(project_name, experiment_name, problem_set, iters; problem_set_batch=problem_set_batch, kwargs...)
end

function run_experiments(args_string::String)
    args = split(args_string)
    run_experiments(args)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_experiments(ARGS);
end
