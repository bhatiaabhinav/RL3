using MDPs
import MDPs: state_space, action_space, action_meaning, state, action, reward, reset!, step!, in_absorbing_state, visualize, to_onehot, factory_reset!
using Random
import DataStructures: CircularBuffer
using ValueIteration
include("vi.jl")

export ValueAugmentedMDP

STATES_MAX_SO_FAR = 0

mutable struct ValueAugmentedMDP{S, A} <: AbstractWrapper{Vector{Float32}, A}
    env::AbstractMDP{S, A}
    laplace_smoothing::Float64
    omit_standard_errors::Bool
    task_horizon::Real
    abstraction_radius::Real
    abstraction_cluster_size::Int
    action_num_bins::Union{Vector{Int}, Nothing}       # number of bins for each action in case of continuous action space
    actions_cumprod_values::Union{Vector{Int}, Nothing} # precomputed cumulative product of number of bins for each action in case of continuous action space
    nactions::Int
    VI_EP::Float64
    Q_DENOM::Float32

    ss::VectorSpace{Float32}
    o::S
    is_start::Bool
    q::Dict{S, Vector{Float32}}
    Eq::Dict{S, Vector{Float32}}
    counts::Dict{S, Vector{Int}}
    model::LearnedTabularModel{S, Int}
    nsteps::Int

    v::Dict{S, Float32}
    Ev::Dict{S, Float32}
    state::Vector{Float32} # obs, q-values, v, Eq, Ev, counts

    prealloc_Q::Union{Matrix{Float64}, Nothing}
    prealloc_V::Union{Vector{Float64}, Nothing}
    prealloc_EQ::Union{Matrix{Float64}, Nothing}
    prealloc_EV::Union{Vector{Float64}, Nothing}
    prealloc_T::Union{Array{Float64, 3}, Nothing}
    prealloc_R::Union{Array{Float64, 3}, Nothing}
    prealloc_ET::Union{Array{Float64, 3}, Nothing}
    prealloc_ER::Union{Array{Float64, 3}, Nothing}

    function ValueAugmentedMDP(env::AbstractMDP{S, A}, laplace_smoothing::Real, omit_standard_errors; task_horizon=Inf, abstraction_radius=0, abstraction_cluster_size=1, action_num_bins=nothing, VI_EP=0.01, Q_DENOM=100.0) where {S, A}
        env_ss = state_space(env)
        env_as = action_space(env)
        m = size(env_ss, 1)
        if A == Int
            actions_cumprod_values = nothing
            n = length(env_as)
        else
            @assert !isnothing(action_num_bins)  "action_num_bins must be provided for continuous action space"
            actions_cumprod_values = [1; cumprod(action_num_bins)[1:end-1]]
            n = prod(action_num_bins)
        end
        env_ss_lows = S == Int ? zeros(Float32, m) : Float32.(env_ss.lows)
        env_ss_highs = S == Int ? ones(Float32, m) : Float32.(env_ss.highs)
        ss = VectorSpace{Float32}(vcat(Float32.(env_ss_lows), -ones(Float32, n + 1), zeros(Float32, n + 1), zeros(Float32, n)), vcat(Float32.(env_ss_highs), ones(Float32, n + 1), ones(Float32, n + 1), ones(Float32, n)))
        nactions = n
        if S == Int
            nstates = m
        else
            nstates = nothing
        end
        if isnothing(nstates)
            model = LearnedTabularModel{S, Int}(;laplace_smoothing=laplace_smoothing,  nactions=nactions, abstraction_radius=abstraction_radius, abstraction_cluster_size=abstraction_cluster_size)
            prealloc_Q = nothing
            prealloc_V = nothing
            prealloc_EQ = nothing
            prealloc_EV = nothing
            prealloc_T = nothing
            prealloc_R = nothing
            prealloc_ET = nothing
            prealloc_ER = nothing
        else
            model = LearnedTabularModel{S, Int}(; laplace_smoothing=laplace_smoothing, nstates=nstates, nactions=nactions, abstraction_radius=abstraction_radius, abstraction_cluster_size=abstraction_cluster_size)
            prealloc_Q = zeros(Float64, nactions, nstates)
            prealloc_V = zeros(Float64, nstates)
            prealloc_EQ = zeros(Float64, nactions, nstates)
            prealloc_EV = zeros(Float64, nstates)
            prealloc_T = zeros(Float64, nstates, nactions, nstates)
            prealloc_R = zeros(Float64, nstates, nactions, nstates)
            prealloc_ET = zeros(Float64, nstates, nactions, nstates)
            prealloc_ER = zeros(Float64, nstates, nactions, nstates)
        end
        new{S, A}(env, laplace_smoothing, omit_standard_errors, task_horizon, abstraction_radius, abstraction_cluster_size, action_num_bins, actions_cumprod_values, nactions, VI_EP, Q_DENOM, ss, state(env), false, Dict{S, Vector{Float32}}(), Dict{S, Vector{Float32}}(), Dict{S, Vector{Int}}(), model, 0, Dict{S, Float32}(), Dict{S, Float32}(), zeros(Float32, m + 3n + 2), prealloc_Q, prealloc_V, prealloc_EQ, prealloc_EV, prealloc_T, prealloc_R, prealloc_ET, prealloc_ER)
    end
end

function factory_reset!(env::ValueAugmentedMDP{S, A}) where {S, A}
    factory_reset!(env.env)
    empty!(env.q)
    empty!(env.v)
    empty!(env.Eq)
    empty!(env.Ev)
    empty!(env.counts)
    env.nsteps = 0
    if S == Int
        nstates = length(state_space(env.env))
    else
        nstates = nothing
    end
    if isnothing(nstates)
        env.model = LearnedTabularModel{S, Int}(; laplace_smoothing=env.laplace_smoothing, nactions=env.nactions, abstraction_radius=env.abstraction_radius, abstraction_cluster_size=env.abstraction_cluster_size)
    else
        env.model = LearnedTabularModel{S, Int}(; laplace_smoothing=env.laplace_smoothing, nstates=nstates, nactions=env.nactions, abstraction_radius=env.abstraction_radius, abstraction_cluster_size=env.abstraction_cluster_size)
    end
    nothing
end

state_space(env::ValueAugmentedMDP) = env.ss
state(env::ValueAugmentedMDP) = env.state

function reset!(env::ValueAugmentedMDP{S, A}; rng::AbstractRNG=Random.GLOBAL_RNG) where {S, A}
    reset!(env.env; rng=rng)
    if S == Int
        env.o = state(env.env)
    else
        copy!(env.o, state(env.env))
    end
    _o = copy(env.o)
    env.is_start = true
    if !haskey(env.q, _o)
        env.q[_o] = zeros(Float32, env.nactions)
        env.counts[_o] = zeros(Int, env.nactions)
        env.v[_o] = 0
        env.Eq[_o] = zeros(Float32, env.nactions)
        env.Ev[_o] = 0
    end
    o = S == Int ? to_onehot(env.o, length(state_space(env.env)), Float32) : convert(Vector{Float32}, env.o)

    o, q, v, Eq, Ev, c = preprocess(o, env.q[env.o], env.v[env.o], env.Eq[env.o], env.Ev[env.o], env.counts[env.o], env.Q_DENOM)
    if env.omit_standard_errors
        Eq .= 0
        Ev = 0
    end
    env.state .= vcat(o, q, v, Eq, Ev, c)
    nothing
end

function discretized_action(env::ValueAugmentedMDP{S, Vector{T}}, a::Vector{T}) where {S, T}
    aspace::VectorSpace{T} = action_space(env.env)
    return discretize(a, aspace.lows, aspace.highs, env.action_num_bins; precomputed_cumprod=env.actions_cumprod_values, assume_inbounds=true)
end

function step!(env::ValueAugmentedMDP{S, A}, a::A; rng=AbstractRNG=Random.GLOBAL_RNG) where {S, A}
    env.nsteps += 1
    _o::S = copy(env.o)
    step!(env.env, a; rng=rng)
    _a::Int = A == Int ? a : discretized_action(env, a)
    env.counts[_o][_a] += 1
    if _a > env.nactions
        error("discretized action $_a is out of bounds. a = $a, nactions = $(env.nactions) _a=$_a")
    end
    r::Float64, o′::S, d::Bool, γ::Float64 = reward(env.env), state(env.env), in_absorbing_state(env.env), 1.0
    _o′::S = copy(o′)
    if !haskey(env.q, _o′)
        env.q[_o′] = zeros(Float32, env.nactions)
        env.counts[_o′] = zeros(Int, env.nactions)
        env.v[_o′] = 0
        env.Eq[_o′] = zeros(Float32, env.nactions)
        env.Ev[_o′] = 0
    end
    update_model!(env.model, env.is_start, _o, _a, r, _o′, d)
    global STATES_MAX_SO_FAR = max(STATES_MAX_SO_FAR, env.model.nstates)
    gamma = 0.95 * γ  # for stability, we reduce gamma
    if isnothing(env.prealloc_Q)
        env.prealloc_Q = zeros(Float64, env.model.nactions, env.model.nstates)
        env.prealloc_V = zeros(Float64, env.model.nstates)
        env.prealloc_EQ = zeros(Float64, env.model.nactions, env.model.nstates)
        env.prealloc_EV = zeros(Float64, env.model.nstates)
        env.prealloc_T = zeros(Float64, env.model.nstates, env.model.nactions, env.model.nstates)
        env.prealloc_R = zeros(Float64, env.model.nstates, env.model.nactions, env.model.nstates)
        env.prealloc_ET = zeros(Float64, env.model.nstates, env.model.nactions, env.model.nstates)
        env.prealloc_ER = zeros(Float64, env.model.nstates, env.model.nactions, env.model.nstates)
        ste_value_iteration!(env.model, gamma, env.task_horizon; q=env.prealloc_Q, v=env.prealloc_V, T=env.prealloc_T, R=env.prealloc_R, E_T=env.prealloc_ET, E_R=env.prealloc_ER, E_q=env.prealloc_EQ, E_v=env.prealloc_EV, ϵ=env.VI_EP, omit_standard_errors=env.omit_standard_errors)
    else
        if size(env.prealloc_Q) == (env.model.nactions, env.model.nstates)
            if env.task_horizon < 100 || !env.omit_standard_errors
                fill!(env.prealloc_Q, 0)
                fill!(env.prealloc_V, 0)
                fill!(env.prealloc_EQ, 0)
                fill!(env.prealloc_EV, 0)
            end
            ste_value_iteration!(env.model, gamma, env.task_horizon; q=env.prealloc_Q, v=env.prealloc_V, T=env.prealloc_T, R=env.prealloc_R, E_T=env.prealloc_ET, E_R=env.prealloc_ER, E_q=env.prealloc_EQ, E_v=env.prealloc_EV, ϵ=env.VI_EP, omit_standard_errors=env.omit_standard_errors)
        else
            # could happen if the num of states increased
            env.prealloc_Q = zeros(Float64, env.model.nactions, env.model.nstates)
            env.prealloc_V = zeros(Float64, env.model.nstates)
            env.prealloc_EQ = zeros(Float64, env.model.nactions, env.model.nstates)
            env.prealloc_EV = zeros(Float64, env.model.nstates)
            env.prealloc_T = zeros(Float64, env.model.nstates, env.model.nactions, env.model.nstates)
            env.prealloc_R = zeros(Float64, env.model.nstates, env.model.nactions, env.model.nstates)
            env.prealloc_ET = zeros(Float64, env.model.nstates, env.model.nactions, env.model.nstates)
            env.prealloc_ER = zeros(Float64, env.model.nstates, env.model.nactions, env.model.nstates)
            ste_value_iteration!(env.model, gamma, env.task_horizon; q=env.prealloc_Q, v=env.prealloc_V, T=env.prealloc_T, R=env.prealloc_R, E_T=env.prealloc_ET, E_R=env.prealloc_ER, E_q=env.prealloc_EQ, E_v=env.prealloc_EV, ϵ=env.VI_EP, omit_standard_errors=env.omit_standard_errors)
        end
    end
    V, Q, EQ, EV = env.prealloc_V, env.prealloc_Q, env.prealloc_EQ, env.prealloc_EV
    o′_id = get_state_id(env.model, _o′)
    env.q[_o′] .= Q[:, o′_id]
    env.v[_o′] = V[o′_id]
    env.Eq[_o′] .= EQ[:, o′_id]
    env.Ev[_o′] = EV[o′_id]

    env.is_start = false

    # ------------------------------------------------------------------

    if S == Int
        env.o = o′
    else
        copy!(env.o, o′)
    end
    o = S == Int ? to_onehot(env.o, length(state_space(env.env)), Float32) : convert(Vector{Float32}, env.o)
    o, q, v, Eq, Ev, c = preprocess(o, env.q[env.o], env.v[env.o], env.Eq[env.o], env.Ev[env.o], env.counts[env.o], env.Q_DENOM)
    if env.omit_standard_errors
        Eq .= 0
        Ev = 0
    end
    env.state .= vcat(o, q, v, Eq, Ev, c)
    nothing
end

function preprocess(o, q, v, Eq, Ev, c, Q_DENOM)
    q = q / Q_DENOM
    v = v / Q_DENOM
    Eq = Eq / Q_DENOM
    Ev = Ev / Q_DENOM
    adv = q .- v  # advantages
    c = c / 10
    return o, adv, v, Eq, Ev, c
end

function visualize(env::ValueAugmentedMDP; kwargs...)
    # update env.v for all states in the model. This if for visualization purposes in gridworlds.
    V = env.prealloc_V
    for id in 1:env.model.nstates
        all_o_with_this_id = get_all_states_with_id(env.model, id)
        for __o in all_o_with_this_id
            env.v[__o] = V[id]
        end
    end
    visualize(env.env; value_fn=env.v, kwargs...)
end
