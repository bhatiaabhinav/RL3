using Revise
using StatsBase
using Random
using MDPs
using MetaMDPs
using BSON
using ImageShow
using Flux
using CUDA
using cuDNN
using BSON
using PPO
using Dates
using Bandits
using Transformers
import EllipsisNotation
import PPO: rollouts_parallel
using Flux: glorot_normal, orthogonal, Recur, zeros32

function import_wandb_if_available()
    try
        @eval import PythonCall
        return Base.invokelatest(PythonCall.pyimport, "wandb")
    catch e
        @warn "wandb not available. Is PythonCall installed? Also, ensure that CondaPkg.toml has wandb in the list of dependencies. Ignore this warning if you don't want to use wandb."
        return nothing
    end
end
wandb = import_wandb_if_available()

const TEST_SEED = 1000000


mutable struct WithIncrementalCaching
    model
    prev_input_shape
    prev_output
end
Flux.@functor WithIncrementalCaching (model, )
function WithIncrementalCaching(model)
    return WithIncrementalCaching(model, (0, 0, 0), nothing)
end
function Flux.reset!(m::WithIncrementalCaching)
    Flux.reset!(m.model)
    m.prev_input_shape = (0, 0, 0)
    Flux.Zygote.@ignore if isa(m.prev_output, CUDA.CuArray); CUDA.unsafe_free!(m.prev_output); end
    m.prev_output = nothing
end
function (m::WithIncrementalCaching)(x)
    D, L, Bs = size(x, 1), size(x, 2), size(x)[3:end]
    prev_D, prev_L, prev_Bs = m.prev_input_shape
    if D != prev_D || L != prev_L + 1 || Bs != prev_Bs
        # @info "Disabling incremental caching" D L Bs prev_D prev_L prev_Bs
        Flux.Zygote.@ignore if isa(m.prev_output, CUDA.CuArray); CUDA.unsafe_free!(m.prev_output); end
        y = m.model(x)
    else
        x_new = selectdim(x, 2, L:L) |> copy
        y_new = m.model(x_new)
        y = cat(m.prev_output, y_new, dims=2)
        Flux.Zygote.@ignore if isa(x_new, CUDA.CuArray); CUDA.unsafe_free!(x_new); end
        Flux.Zygote.@ignore if isa(y_new, CUDA.CuArray); CUDA.unsafe_free!(y_new); end
    end
    m.prev_input_shape = (D, L, Bs)
    Flux.Zygote.@ignore if isa(m.prev_output, CUDA.CuArray); CUDA.unsafe_free!(m.prev_output); end
    m.prev_output = y
    return y
end

function make_random_mdps(; seed=0, nstates=10, nactions=5, task_horizon=10, horizon=100, gamma=1, ood=false, random_mdps_dirichlet_alpha=1.0, random_mdps_rewards_std=1.0, kwargs...)
    m, n = nstates, nactions
    TH = task_horizon
    H = horizon
    if ood
        random_mdps_dirichlet_alpha = 0.25
    end
    mdps = MDPGenerator((i, rng) -> RandomDiscreteMDP(rng, m, n; uniform_dist_rewards=false, α=random_mdps_dirichlet_alpha, β=random_mdps_rewards_std), Xoshiro(seed))
    seed = seed > 0 ? seed - 1 : seed
    mdps_test = MDPGenerator((i, rng) -> RandomDiscreteMDP(Xoshiro(TEST_SEED+seed+i-1), m, n; uniform_dist_rewards=false, α=random_mdps_dirichlet_alpha, β=random_mdps_rewards_std), Random.GLOBAL_RNG)  # mdps will have seeds TEST_SEED, TEST_SEED+1, TEST_SEED+2, ... over meta-episodes. For batch testing, which will be 1 meta-episode per batch, seeds are TEST_SEED, TEST_SEED+1, TEST_SEED+2, ... for 1st mdp in the respective batch. This assumes that the main mdp has seed 0, and batch mdps have seeds 1, 2, 3, ...
    γ = gamma
    sspace = state_space(mdps)
    aspace = action_space(mdps)
    m, n = size(sspace, 1), size(aspace, 1)
    return mdps, mdps_test, TH, γ, H, sspace, aspace, m, n
end

function make_bandits(; seed=0, narms=5, horizon=100, gamma=1, ood=false, kwargs...)
    k = narms
    H, TH = horizon, 1
    if ood
        # use normal distribution (mean = 0.5, std = 0.5) to generate bandit success probabilities
        mdps = MDPGenerator((i, rng) -> BernauliMultiArmedBandit(randn(rng, k) .* 0.5 .+ 0.5), Xoshiro(seed))
        seed = seed > 0 ? seed - 1 : seed
        mdps_test = MDPGenerator((i, rng) -> BernauliMultiArmedBandit(randn(Xoshiro(TEST_SEED+seed+i-1), k) .* 0.5 .+ 0.5), Random.GLOBAL_RNG)
    else
        # use uniform distribution to generate bandit success probabilities
        mdps = MDPGenerator((i, rng) -> BernauliMultiArmedBandit(rand(rng, k)), Xoshiro(seed))
        seed = seed > 0 ? seed - 1 : seed
        mdps_test = MDPGenerator((i, rng) -> BernauliMultiArmedBandit(rand(Xoshiro(TEST_SEED+seed+i-1), k)), Random.GLOBAL_RNG)
    end
    γ = gamma
    sspace = state_space(mdps)
    aspace = action_space(mdps)
    m, n = size(sspace, 1), size(aspace, 1)
    # @info "created random bandits" TH, γ, H, m, n
    return mdps, mdps_test, TH, γ, H, sspace, aspace, m, n
end

function make_gridworlds(; seed=0, horizon=200, task_horizon=horizon, gamma=1, variation="11x11", kwargs...)
    H, TH = horizon, task_horizon
    variation_dict = Dict(
        "11x11" => CUSTOMGW_11x11_PARAMS,
        "11x11_deterministic" => CUSTOMGW_11x11_PARAMS_DETERMINISTIC,
        "13x13" => CUSTOMGW_13x13_PARAMS,
        "13x13_dense" => CUSTOMGW_13x13_PARAMS_DENSE,
        "13x13_deterministic" => CUSTOMGW_13x13_PARAMS_DETERMINISTIC,
        "13x13_watery" => CUSTOMGW_13x13_PARAMS_WATERY,
        "13x13_dangerous" => CUSTOMGW_13x13_PARAMS_DANGEROUS,
        "13x13_corner" => CUSTOMGW_13x13_PARAMS_CORNER,
    )
    grid_args = variation_dict[variation]
    mdps = MDPGenerator((i, rng) -> GridWorldContinuous{Float32}(CustomGridWorld(rng, grid_args)), Xoshiro(seed))
    seed = seed > 0 ? seed - 1 : seed
    mdps_test = MDPGenerator((i, rng) -> GridWorldContinuous{Float32}(CustomGridWorld(Xoshiro(TEST_SEED+seed+i-1), grid_args)), Random.GLOBAL_RNG)
    γ = gamma
    sspace = state_space(mdps)
    aspace = action_space(mdps)
    m, n = size(sspace, 1), size(aspace, 1)
    return mdps, mdps_test, TH, γ, H, sspace, aspace, m, n
end


"""Returns another problem"""
function wrap_onehot_mdps(problem)
    (mdps, mdps_test, TH, γ, H, sspace, aspace, m, n) = problem
    mdps = Iterators.map(OneHotStateReprWrapper{Float32}, mdps);
    mdps_test = Iterators.map(OneHotStateReprWrapper{Float32}, mdps_test);
    mdp1, _ = iterate(mdps)
    sspace, aspace = state_space(mdp1), action_space(mdp1);
    m, n = size(sspace, 1), size(aspace, 1)
    # @info "wrapped OneHotStateRepr" TH, γ, H, m, n
    return mdps, mdps_test, TH, γ, H, sspace, aspace, m, n
end

"""Wrap value augmented mdps and return another problem"""
function wrap_VAMDPs(problem; laplace_smoothing, omit_standard_errors, task_horizon=Inf, abstraction_radius=0, abstraction_cluster_size=1, action_num_bins=nothing, Q_DENOM, VI_EP)
    # println("action_num_bins: $action_num_bins")
    (mdps, mdps_test, TH, γ, H, sspace, aspace, m, n) = problem
    mdps = Iterators.map(m -> ValueAugmentedMDP(m, laplace_smoothing, omit_standard_errors; task_horizon=task_horizon, abstraction_radius=abstraction_radius, abstraction_cluster_size=abstraction_cluster_size, action_num_bins=action_num_bins, Q_DENOM=Q_DENOM, VI_EP=VI_EP), mdps);
    mdps_test = Iterators.map(m -> ValueAugmentedMDP(m, laplace_smoothing, omit_standard_errors; task_horizon=task_horizon, abstraction_radius=abstraction_radius, abstraction_cluster_size=abstraction_cluster_size, action_num_bins=action_num_bins, Q_DENOM=Q_DENOM, VI_EP=VI_EP), mdps_test);
    mdp1, _ = iterate(mdps)
    sspace, aspace = state_space(mdp1), action_space(mdp1);
    m, n = size(sspace, 1), size(aspace, 1);
    # @info "wrapped VA wrapper" TH, γ, H, m, n
    return mdps, mdps_test, TH, γ, H, sspace, aspace, m, n
end


function make_metamdp(problem; include_time_context)
    (mdps, mdps_test, TH, γ, H, sspace, aspace, m, n) = problem
    metamdp, metamdp_test = MetaMDP(mdps, H, include_time_context; task_horizon=TH), MetaMDP(mdps_test, H, include_time_context; task_horizon=TH)
    sspace, aspace = state_space(metamdp), action_space(metamdp)
    m, n = size(sspace, 1), size(aspace, 1);
    return metamdp, metamdp_test, (mdps, mdps_test, TH, γ, H, sspace, aspace, m, n)
end

function test_random_policy(problem; test_episodes=1000, kwargs...)
    (mdps, mdps_test, TH, γ, H, sspace, aspace, m, n) = problem
    metamdp, metamdp_test = make_metamdp(problem; include_time_context=:none)
    score = interact(metamdp_test, RandomPolicy(metamdp_test), γ, H, test_episodes, ProgressMeterHook(; desc="Evaluating Random Policy"); rng=Xoshiro(TEST_SEED))[1] |> mean
    return score, [], []
end

function loadmodels(filename)
    BSON.@load filename models
    return models
end
function save_models(models, dirname, eps, steps)
    if eps == 1
        rm(dirname, recursive=true, force=true)
    end
    mkpath(dirname)
    models = cpu(models)
    BSON.@save "$(dirname)/ep-$(eps)-steps-$(steps).bson" models
end

function do_ppo_learning(project_name, experiment_name, problem_set, iters; problem_set_batch, model, dmodel, lr, log_interval, model_save_interval, nsteps, nepochs, ent_bonus, kl_target, ppo_epsilon, lambda, seed,  advantagenorm, device, inference_device, adam_eps, adam_wd, clipnorm, minibatch_size, progressmeter, iters_per_postepisode, video, video_interval, act_greedy, nheads, ndecoders, test_model, continue_model, decay_ent_bonus, decay_lr, config, problem_name, enable_wandb, obsnorm, rewardnorm, no_multithreading, no_plots, test_episodes, include_time_context, no_pe, no_decoder, no_evidence_wrapper, parallel_testing, algo, kwargs...)
    metamdp, metamdp_test, meta_problem = make_metamdp(problem_set; include_time_context=include_time_context)
    (mdps, mdps_test, TH, γ, H, sspace, aspace, m, n) = meta_problem
    @info "created meta_mdp" m, n
    T, Tₐ = Float32, eltype(eltype(aspace))

    norm_by_rew = false

    if !no_evidence_wrapper
        metamdp = EvidenceObservationWrapper{T}(metamdp)
        metamdp_test = EvidenceObservationWrapper{T}(metamdp_test)
    end

    if obsnorm || rewardnorm
        metamdp = NormalizeWrapper(metamdp, normalize_reward=rewardnorm, normalize_obs=obsnorm, normalize_reward_by_reward_std=norm_by_rew)
        metamdp.update_stats = false
        if continue_model == ""
            obs_rmv, ret_rmv, rew_rmv = metamdp.obs_rmv, metamdp.ret_rmv, metamdp.rew_rmv
        else
            _, _, obs_rmv, ret_rmv, rew_rmv = loadmodels(continue_model)
            metamdp.obs_rmv, metamdp.ret_rmv, metamdp.rew_rmv = obs_rmv, ret_rmv, rew_rmv
        end
        metamdp_test = NormalizeWrapper(metamdp_test, obs_rmv=obs_rmv, ret_rmv=ret_rmv, rew_rmv=rew_rmv, normalize_reward=rewardnorm, normalize_obs=obsnorm, normalize_reward_by_reward_std=norm_by_rew)
        metamdp_test.update_stats = false
    end

    get_reward_multipler_fn() = rewardnorm ? (norm_by_rew ? std(rew_rmv) : std(ret_rmv)) : 1.0

    name = experiment_name
    PPOActor = aspace isa IntegerSpace ? PPOActorDiscrete{T} : PPOActorContinuous{T, Tₐ}
    recur_type = model == "transformer" ? TRANSFORMER : model == "rnn" ? RECURRENT : MARKOV

    m, n = size(state_space(metamdp), 1), size(action_space(metamdp), 1)
    if test_model == ""
        if continue_model == ""
            if model == "transformer"
                dim_k = dmodel ÷ nheads
                dim_v = dim_k
                dim_ff = 4 * dmodel
                MakeProjectToDimModelMDP() = Chain(Dense(m, dim_ff, relu), Dense(dim_ff, dmodel, relu))
                MakeProjectToDimModel = MakeProjectToDimModelMDP
                MakePositionalEncoder(incremental_inference_mode) = no_pe ? identity : LearnedPositionalEncoder(dmodel, H+1; incremental_inference_mode=incremental_inference_mode)
                MakeDecoder(incremental_inference_mode) = no_decoder ? identity : Decoder(dmodel, dim_k, dim_v, nheads, dim_ff, ndecoders; dropout=false, no_encoder=true, incremental_inference_mode=incremental_inference_mode)
                MakePreFinalLayer() = no_decoder ? Dense(dmodel, dmodel, relu) : relu  # relu required since the final layer in the decoder is a linear layer with no activation
                _actor_model = WithIncrementalCaching(
                    Chain(
                        MakeProjectToDimModel(),
                        LayerNorm(dmodel, affine=true),
                        MakePositionalEncoder(true),
                        MakeDecoder(true),
                        MakePreFinalLayer(),
                        Dense(dmodel, n)
                    )
                )
                critic_model = Chain(
                    MakeProjectToDimModel(),
                    LayerNorm(dmodel, affine=true),
                    MakePositionalEncoder(false),
                    MakeDecoder(false),
                    MakePreFinalLayer(),
                    Dense(dmodel, 1)
                )
            elseif model == "rnn"
                _actor_model = Chain(Dense(m, dmodel, relu), GRUv3(dmodel, 4*dmodel), Dense(4*dmodel, n))
                critic_model = Chain(Dense(m, dmodel, relu), GRUv3(dmodel, 4*dmodel), Dense(4*dmodel, 1))
            elseif model == "markov"
                dim_ff = 4 * dmodel
                _actor_model = Chain(Dense(m, dim_ff, relu), Dense(dim_ff, dim_ff, relu), Dense(dim_ff, n))
                critic_model = Chain(Dense(m, dim_ff, relu), Dense(dim_ff, dim_ff, relu), Dense(dim_ff, 1))
            else
                error("What's a $model model?")
            end
        else
            if obsnorm || rewardnorm
                _actor_model, critic_model, _, _, _ = loadmodels(continue_model)
            else
                _actor_model, critic_model = loadmodels(continue_model)
            end
            @info "Loaded models"
            if _actor_model isa WithIncrementalCaching
                _actor_model = WithIncrementalCaching(_actor_model.model)
            end
        end
        println("Actor model: ")
        display(_actor_model isa WithIncrementalCaching ? _actor_model.model : _actor_model)
        println("Critic model: ")
        display(critic_model)

        actor_model = _actor_model |> inference_device
        p = PPOActor(actor_model, false, aspace, recur_type)
        gp = PPOActor(actor_model, true, aspace, recur_type)


        metamdps_batch = map(problem_set_batch) do _pb
            _metamdp = make_metamdp(_pb; include_time_context=include_time_context)[1]
            if !no_evidence_wrapper
                _metamdp = EvidenceObservationWrapper{T}(_metamdp)
            end
            _metamdp = obsnorm || rewardnorm ? NormalizeWrapper(_metamdp, obs_rmv=obs_rmv, ret_rmv=ret_rmv, rew_rmv=rew_rmv, normalize_reward=rewardnorm, normalize_obs=obsnorm, normalize_reward_by_reward_std=norm_by_rew) : _metamdp
            return _metamdp
        end
        metamdps_batch = VecEnv(metamdps_batch, !no_multithreading)
        ppolh = PPOLearner(envs = metamdps_batch, actor=p, critic=critic_model, nsteps=nsteps, batch_size = minibatch_size, nepochs=nepochs, entropy_bonus=ent_bonus, decay_ent_bonus=decay_ent_bonus, clipnorm=clipnorm, normalize_advantages = advantagenorm, lr_critic=lr, lr_actor=lr, decay_lr=decay_lr, min_lr=1f-5, device=device, ppo=true, kl_target=kl_target, ϵ=ppo_epsilon, λ=lambda, adam_epsilon=adam_eps, adam_weight_decay=adam_wd, early_stop_critic=false, iters_per_postepisode=iters_per_postepisode, progressmeter=progressmeter)

        get_stats() = Dict(ppolh.stats..., :reward_multiplier => get_reward_multipler_fn())
        lh = LoggingHook(get_stats; smooth_over=1000)
        drh = DataRecorderHook(get_stats, "data/$project_name/$name.csv", overwrite=false)
        ph = no_plots ? EmptyHook() : PlotEverythingHook("data/$project_name", "plots/$project_name")
        video_dir = "videos/$project_name/$name"
        wh = enable_wandb ? WandbHook(wandb, project_name, name, config=config, stats_getter=get_stats, video_file_getter=video ? monitor_video_dir(video_dir) : nothing, smooth_over=1000) : EmptyHook()
        vrh = video ? VideoRecorderHook(video_dir, ceil(Int, video_interval / iters_per_postepisode); vmax=100) : EmptyHook()
        models_to_save = obsnorm || rewardnorm ? (actor_model, critic_model, obs_rmv, ret_rmv, rew_rmv) : (actor_model, critic_model)
        msh = EveryNEpisodesHook((; returns, steps, kwargs...) -> save_models(models_to_save, "models/$project_name/$name", length(returns), steps), model_save_interval ÷ iters_per_postepisode)
        act_policy = act_greedy ? gp : p
        rs, ls = interact(metamdp, act_policy, γ, H, iters ÷ iters_per_postepisode,
            ppolh,
            lh,
            drh,
            ph,
            vrh,
            wh,
            msh,
            GCHook(),
            ProgressMeterHook(),
            SleepHook(0.01);
            rng=Xoshiro(seed), reward_multiplier=get_reward_multipler_fn)
    else
        if obsnorm || rewardnorm
            _actor_model, _critic_model, obs_rmv, ret_rmv, rew_rmv = loadmodels(test_model)
        else
            _actor_model, _critic_model = loadmodels(test_model)
        end
        @info "Loaded models"
        _actor_model = model == "transformer" ? WithIncrementalCaching(_actor_model.model) : _actor_model
        println("Actor model: ")
        display(_actor_model isa WithIncrementalCaching ? _actor_model.model : _actor_model)
        println("Critic model: ")
        display(_critic_model)

        testing_device = parallel_testing ? device : inference_device
        actor_model = _actor_model |> testing_device
        p = PPOActor(actor_model, false, aspace, recur_type)
        gp = PPOActor(actor_model, true, aspace, recur_type)
    end

    println("Testing policy. Greedy=$(act_greedy)")
    test_policy = act_greedy ? gp : p

    if parallel_testing
        if video
            @error "Parallel testing does not support video recording."
        end
        println("Parallel testing nenvs = $(length(problem_set_batch))")
        metamdps_batch_test = map(problem_set_batch) do _pb
            _metamdp_test = make_metamdp(_pb; include_time_context=include_time_context)[2]
            if !no_evidence_wrapper
                _metamdp_test = EvidenceObservationWrapper{T}(_metamdp_test)
            end
            if obsnorm || rewardnorm
                _metamdp_test = NormalizeWrapper(_metamdp_test, obs_rmv=obs_rmv, ret_rmv=ret_rmv, rew_rmv=rew_rmv, normalize_reward=rewardnorm, normalize_obs=obsnorm, normalize_reward_by_reward_std=norm_by_rew)
                _metamdp_test.update_stats = false
            end
            return _metamdp_test
        end
        metamdps_batch_test_venv = VecEnv(metamdps_batch_test, !no_multithreading)
        factory_reset!(metamdps_batch_test_venv)
        if H <= 1024
            _, _, 𝐫, _, _ = rollouts_parallel(metamdps_batch_test_venv, H, test_policy, testing_device, Xoshiro(TEST_SEED), true)
        elseif H <= 2048
            # split batch into two halves
            half_batch_size = div(length(metamdps_batch_test), 2)
            metamdps_batch_test_venv_1 = VecEnv(metamdps_batch_test[1:half_batch_size], !no_multithreading)
            metamdps_batch_test_venv_2 = VecEnv(metamdps_batch_test[half_batch_size+1:end], !no_multithreading)
            println("First half")
            _, _, 𝐫1, _, _ = rollouts_parallel(metamdps_batch_test_venv_1, H, test_policy, testing_device, Xoshiro(TEST_SEED), true)
            println("Second half")
            _, _, 𝐫2, _, _ = rollouts_parallel(metamdps_batch_test_venv_2, H, test_policy, testing_device, Xoshiro(TEST_SEED), true)
            𝐫 = cat(𝐫1, 𝐫2, dims=3)
        else
            # 4 parts:
            one_fourth_batch_size = div(length(metamdps_batch_test), 4)
            metamdps_batch_test_venv_1 = VecEnv(metamdps_batch_test[1:one_fourth_batch_size], !no_multithreading)
            metamdps_batch_test_venv_2 = VecEnv(metamdps_batch_test[one_fourth_batch_size+1:2*one_fourth_batch_size], !no_multithreading)
            metamdps_batch_test_venv_3 = VecEnv(metamdps_batch_test[2*one_fourth_batch_size+1:3*one_fourth_batch_size], !no_multithreading)
            metamdps_batch_test_venv_4 = VecEnv(metamdps_batch_test[3*one_fourth_batch_size+1:end], !no_multithreading)
            println("First quarter")
            _, _, 𝐫1, _, _ = rollouts_parallel(metamdps_batch_test_venv_1, H, test_policy, testing_device, Xoshiro(TEST_SEED), true)
            println("Second quarter")
            _, _, 𝐫2, _, _ = rollouts_parallel(metamdps_batch_test_venv_2, H, test_policy, testing_device, Xoshiro(TEST_SEED), true)
            println("Third quarter")
            _, _, 𝐫3, _, _ = rollouts_parallel(metamdps_batch_test_venv_3, H, test_policy, testing_device, Xoshiro(TEST_SEED), true)
            println("Fourth quarter")
            _, _, 𝐫4, _, _ = rollouts_parallel(metamdps_batch_test_venv_4, H, test_policy, testing_device, Xoshiro(TEST_SEED), true)
            𝐫 = cat(𝐫1, 𝐫2, 𝐫3, 𝐫4, dims=3)
        end
        Rs = sum(𝐫, dims=2)[:]
    else
        vrh = video ? VideoRecorderHook("videos/$project_name/$name-test", video_interval; vmax=100) : EmptyHook()
        Rs = interact(metamdp_test, test_policy, γ, H, test_episodes, ProgressMeterHook(), vrh, GCHook(), SleepHook(0.01); rng=Xoshiro(TEST_SEED), reward_multiplier=get_reward_multipler_fn)[1];
    end

    score = mean(Rs)
    score_std = std(Rs)
    score_ste = score_std / sqrt(length(Rs))
 
    println("Final score ", score, "±", score_ste, " (std ", score_std, ")", " (n=", length(Rs), ")")

    return score
end
