using Revise
using StatsBase
using Random
using MDPs
using MetaMDPs
using Flux
using CUDA
using BSON
using PPO
using Dates
using Bandits
using MDPVideoRecorder
using Transformers
using Flux: glorot_normal, orthogonal, Recur, zeros32

const TEST_SEED = 1

function make_random_mdps(; seed=0, nstates=10, nactions=5, task_horizon=10, horizon=100, gamma=1, ood=false, random_mdps_dirichlet_alpha=1.0, random_mdps_rewards_std=1.0, kwargs...)
    m, n = nstates, nactions
    TH = task_horizon
    H = horizon
    mdps = MDPGenerator((i, rng) -> RandomDiscreteMDP(rng, m, n; uniform_dist_rewards=ood, α=random_mdps_dirichlet_alpha, β=random_mdps_rewards_std), Xoshiro(seed))
    mdps_test = MDPGenerator((i, rng) -> RandomDiscreteMDP(rng, m, n; uniform_dist_rewards=ood, α=random_mdps_dirichlet_alpha, β=random_mdps_rewards_std), Xoshiro(TEST_SEED))
    γ = gamma
    sspace = state_space(mdps)
    aspace = action_space(mdps)
    m = mdps |> state_space |> length
    n = mdps |> action_space |> length
    # @info "created random mdps" TH, γ, H, m, n
    return mdps, mdps_test, TH, γ, H, sspace, aspace, m, n
end

function make_bandits(; seed=0, narms=5, horizon=100, gamma=1, ood=false, kwargs...)
    k = narms
    H, TH = horizon, 1
    if ood
        # use normal distribution (mean = 0.5, std = 0.5) to generate bandit success probabilities
        mdps = MDPGenerator((i, rng) -> BernauliMultiArmedBandit(randn(rng, k) .* 0.5 .+ 0.5), Xoshiro(seed))
        mdps_test = MDPGenerator((i, rng) -> BernauliMultiArmedBandit(randn(rng, k) .* 0.5 .+ 0.5), Xoshiro(TEST_SEED))
    else
        # use uniform distribution to generate bandit success probabilities
        mdps = MDPGenerator((i, rng) -> BernauliMultiArmedBandit(rand(rng, k)), Xoshiro(seed))
        mdps_test = MDPGenerator((i, rng) -> BernauliMultiArmedBandit(rand(rng, k)), Xoshiro(TEST_SEED))
    end
    γ = gamma
    sspace = state_space(mdps)
    aspace = action_space(mdps)
    m = mdps |> state_space |> length
    n = mdps |> action_space |> length
    # @info "created random bandits" TH, γ, H, m, n
    return mdps, mdps_test, TH, γ, H, sspace, aspace, m, n
end

function make_gridworlds(; seed=0, horizon=200, task_horizon=horizon, gamma=1, grid_variation="11x11", kwargs...)
    H, TH = horizon, task_horizon
    mdps = MDPGenerator((i, rng) -> GridWorldContinuous{Float32}(GridWorld(make_grid(rng, grid_variation), enter_rewards, failuremode_slip_probability=slip_probabality, absorbing_states=absorbing_states)), Xoshiro(seed))
    mdps_test = MDPGenerator((i, rng) -> GridWorldContinuous{Float32}(GridWorld(make_grid(rng, grid_variation), enter_rewards, failuremode_slip_probability=slip_probabality, absorbing_states=absorbing_states)), Xoshiro(TEST_SEED))
    γ = gamma
    sspace = state_space(mdps)
    aspace = action_space(mdps)
    m = mdps |> state_space |> length
    n = mdps |> action_space |> length
    return mdps, mdps_test, TH, γ, H, sspace, aspace, m, n
end


"""Returns another problem"""
function wrap_onehot_mdps(problem)
    (mdps, mdps_test, TH, γ, H, sspace, aspace, m, n) = problem
    mdps = Iterators.map(OneHotStateReprWrapper{Float32}, mdps);
    mdps_test = Iterators.map(OneHotStateReprWrapper{Float32}, mdps_test);
    mdp1, _ = iterate(mdps)
    sspace, aspace = state_space(mdp1), action_space(mdp1);
    m, n = size(sspace, 1), length(aspace);
    # @info "wrapped OneHotStateRepr" TH, γ, H, m, n
    return mdps, mdps_test, TH, γ, H, sspace, aspace, m, n
end

"""Wrap value augmented mdps and return another problem"""
function wrap_VAMDPs(problem; task_horizon=Inf, abstraction_radius=0, abstraction_cluster_size=1, Q_DENOM, VI_EP, drop_observation=false)
    (mdps, mdps_test, TH, γ, H, sspace, aspace, m, n) = problem
    mdps = Iterators.map(m -> ValueEstimateAugmentedMDPER(m; task_horizon=task_horizon, abstraction_radius=abstraction_radius, abstraction_cluster_size=abstraction_cluster_size, Q_DENOM=Q_DENOM, VI_EP=VI_EP, drop_observation=drop_observation), mdps);
    mdps_test = Iterators.map(m -> ValueEstimateAugmentedMDPER(m; task_horizon=task_horizon, abstraction_radius=abstraction_radius, abstraction_cluster_size=abstraction_cluster_size, Q_DENOM=Q_DENOM, VI_EP=VI_EP, drop_observation=drop_observation), mdps_test);
    mdp1, _ = iterate(mdps)
    sspace, aspace = state_space(mdp1), action_space(mdp1);
    m, n = size(sspace, 1), length(aspace);
    # @info "wrapped VA wrapper" TH, γ, H, m, n
    return mdps, mdps_test, TH, γ, H, sspace, aspace, m, n
end

function make_metamdp(problem; add_time_context=true)
    (mdps, mdps_test, TH, γ, H, sspace, aspace, m, n) = problem
    if add_time_context
        metamdp, metamdp_test = MetaMDPwithTimeContext(mdps, H; task_horizon=TH), MetaMDPwithTimeContext(mdps_test, H; task_horizon=TH)
    else
        metamdp, metamdp_test = MetaMDP(mdps; task_horizon=TH), MetaMDP(mdps_test; task_horizon=TH)
    end
    sspace, aspace = state_space(metamdp), action_space(metamdp)
    m, n = size(sspace, 1), length(aspace);
    return metamdp, metamdp_test, (mdps, mdps_test, TH, γ, H, sspace, aspace, m, n)
end

function test_random_policy(problem; add_time_context=false, seed=0, kwargs...)
    (mdps, mdps_test, TH, γ, H, sspace, aspace, m, n) = problem
    metamdp, metamdp_test = make_metamdp(problem; add_time_context=add_time_context)
    score = interact(metamdp_test, RandomPolicy(metamdp_test), γ, H, 1000; rng=Xoshiro(seed))[1] |> mean
    return score, [], []
end


function do_ppo_learning(experiment_name, problem, iters; problem_batch, model, dmodel, lr, log_interval, model_save_interval, nsteps, nepochs, ent_bonus, kl_target, ppo_epsilon, lambda, seed,  advantagenorm, device, adam_eps, adam_wd, clipnorm, minibatch_size, progressmeter, video, video_interval, act_greedy, nheads, ndecoders, test_model, extrapolate_from, decay_ent_bonus, decay_lr, config, kwargs...)
    metamdp, metamdp_test, meta_problem = make_metamdp(problem; add_time_context=true)
    (mdps, mdps_test, TH, γ, H, sspace, aspace, m, n) = meta_problem
    @info "created meta_mdp" m, n
    T = Float32

    metamdp = EvidenceObservationWrapper{T}(metamdp)
    metamdp_test = EvidenceObservationWrapper{T}(metamdp_test)
    name = experiment_name

    if test_model == ""
        if model == "transformer"
            dim_k = dmodel ÷ nheads
            dim_v = dim_k
            dim_ff = 4 * dmodel
            embed_layer_actor = Dense(n+1+1+m, dmodel, bias=false)
            embed_layer_actor.weight .*= Float32(sqrt(dmodel))
            pe_layer_actor = LearnedPositionalEncoder(dmodel, 1024)
            actor_model = Chain(embed_layer_actor, LayerNorm(dmodel), pe_layer_actor, Decoder(dmodel, dim_k, dim_v, nheads, dim_ff, ndecoders; no_encoder=true, dropout=false), Dense(dmodel, n))
            embed_layer_critic = Dense(n+1+1+m, dmodel, bias=false)
            embed_layer_critic.weight .*= Float32(sqrt(dmodel))
            pe_layer_critic = LearnedPositionalEncoder(dmodel, 1024)
            critic_model = Chain(embed_layer_critic, LayerNorm(dmodel), pe_layer_critic, Decoder(dmodel, dim_k, dim_v, nheads, dim_ff, ndecoders; dropout=false, no_encoder=true), Dense(dmodel, 1))
            p = PPOActorDiscrete{T}(actor_model, false, aspace, TRANSFORMER)
            gp = PPOActorDiscrete{T}(actor_model, true, aspace, TRANSFORMER)
        elseif model == "rnn"
            actor_model = Chain(Dense(n+1+1+m, dmodel, relu), GRUv3(dmodel, 4*dmodel), Dense(4*dmodel, n))
            critic_model = Chain(Dense(n+1+1+m, dmodel, relu), GRUv3(dmodel, 4*dmodel), Dense(4*dmodel, 1))
            p = PPOActorDiscrete{T}(actor_model, false, aspace, RECURRENT)
            gp = PPOActorDiscrete{T}(actor_model, true, aspace, RECURRENT)
        elseif model == "markov"
            actor_model = Chain(Dense(n+1+1+m, dmodel, relu), Dense(dmodel, dmodel, relu), Dense(dmodel, n))
            critic_model = Chain(Dense(n+1+1+m, dmodel, relu), Dense(dmodel, dmodel, relu), Dense(dmodel, 1))
            p = PPOActorDiscrete{T}(actor_model, false, aspace, MARKOV)
            gp = PPOActorDiscrete{T}(actor_model, true, aspace, MARKOV)
        else
            error("What's a $model?")
        end

        metamdps_batch = [make_metamdp(pb; add_time_context=true)[1] for pb in problem_batch]
        metamdps_batch = [EvidenceObservationWrapper{T}(mdp) for mdp in metamdps_batch]
        ppol = PPOLearner(envs = metamdps_batch, actor=p, critic=critic_model, nsteps=nsteps, trajs_per_minibatch = minibatch_size ÷ nsteps, nepochs=nepochs, entropy_bonus=ent_bonus, decay_ent_bonus=decay_ent_bonus, clipnorm=clipnorm, normalize_advantages = advantagenorm, lr_critic=lr, lr_actor=lr, decay_lr=decay_lr, device=device, ppo=true, kl_target=kl_target, ϵ=ppo_epsilon, λ=lambda, adam_epsilon=adam_eps, adam_weight_decay=adam_wd, early_stop_critic=false, progressmeter=progressmeter)

        get_stats() = ppol.stats

        act_policy = act_greedy ? gp : p
        if video
            dirname = "videos/$name"
            rm(dirname, recursive=true, force=true)
            mkpath(dirname)
            rs, ls = interact(metamdp, act_policy, γ, H, iters, ppol, StatsPrintHook(get_stats, log_interval), ProgressMeterHook(), ModelsSaveHook((actor_model, critic_model), name, model_save_interval), VideoRecorder(dirname, "mp4"; interval=video_interval), GCHook(); rng=Xoshiro(seed), vmax=100);
        else
            rs, ls = interact(metamdp, act_policy, γ, H, iters, ppol, StatsPrintHook(get_stats,  log_interval), ProgressMeterHook(), ModelsSaveHook((actor_model, critic_model), name, model_save_interval), GCHook(); rng=Xoshiro(seed));
        end

        iters = 1000 # for Testing
    else
        actor_model, critic_model = loadmodels(test_model)
        
        @info "Loaded models"
        if model == "transformer"
            p = PPOActorDiscrete{T}(actor_model, false, aspace, TRANSFORMER)
            gp = PPOActorDiscrete{T}(actor_model, true, aspace, TRANSFORMER)
        elseif model == "rnn"
            p = PPOActorDiscrete{T}(actor_model, false, aspace, RECURRENT)
            gp = PPOActorDiscrete{T}(actor_model, true, aspace, RECURRENT)
        elseif model == "markov"
            p = PPOActorDiscrete{T}(actor_model, false, aspace, MARKOV)
            gp = PPOActorDiscrete{T}(actor_model, true, aspace, MARKOV)
        else
            error("What's a $model?")
        end 
    end

    println("Testing policy. Greedy=$(act_greedy)")
    test_policy = act_greedy ? gp : p
    metamdp_test.update_stats = false
    if video
        dirname = "finalvids/$name-greedy-$act_greedy"
        rm(dirname, recursive=true, force=true)
        mkpath(dirname)
        Rs = interact(metamdp_test, test_policy, γ, H, iters, ProgressMeterHook(), VideoRecorder(dirname, "mp4"; interval=video_interval), GCHook(); rng=Xoshiro(seed), vmax=100)[1];
        score = mean(Rs)
        score_std = std(Rs)
        score_ste = score_std / sqrt(length(Rs))
    else
        Rs = interact(metamdp_test, test_policy, γ, H, iters, ProgressMeterHook(), GCHook(); rng=Xoshiro(seed))[1];
        score = mean(Rs)
        score_std = std(Rs)
        score_ste = score_std / sqrt(length(Rs))
    end
 
    println("Final score ", score, "±", score_ste, " (std ", score_std, ")")

    return score
end
