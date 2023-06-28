using MDPs
import MDPs: state_space, action_space, action_meaning, state, action, reward, reset!, step!, in_absorbing_state, visualize, to_onehot, factory_reset!
using Random
import DataStructures: CircularBuffer
using ValueIteration

export ValueEstimateAugmentedMDPER

mutable struct ValueEstimateAugmentedMDPER{S} <: AbstractMDP{Vector{Float32}, Int}
    env::AbstractMDP{S, Int}
    α::Float32
    drop_observation::Bool
    task_horizon::Real
    abstraction_radius::Real
    abstraction_cluster_size::Int
    VI_EP::Float64
    Q_DENOM::Float32

    ss::VectorSpace{Float32}
    o::S
    is_start::Bool
    q::Dict{S, Vector{Float32}}
    counts::Dict{S, Vector{Int}}
    model::LearnedTabularModel{S, Int}

    v::Dict{S, Float32}
    state::Vector{Float32} # obs, q-values

    prealloc_Q::Union{Matrix{Float64}, Nothing}
    prealloc_V::Union{Vector{Float64}, Nothing}
    prealloc_T::Union{Array{Float64, 3}, Nothing}
    prealloc_R::Union{Array{Float64, 3}, Nothing}

    function ValueEstimateAugmentedMDPER(env::AbstractMDP{S, Int}; drop_observation::Bool=false, task_horizon=Inf, abstraction_radius=0, abstraction_cluster_size=1, VI_EP=0.01, Q_DENOM=100.0) where {S}
        env_ss = state_space(env)
        env_as = action_space(env)
        m, n = size(env_ss, 1), size(env_as, 1)
        env_ss_lows = S == Int ? zeros(Float32, m) : Float32.(env_ss.lows)
        env_ss_highs = S == Int ? ones(Float32, m) : Float32.(env_ss.highs)
        if drop_observation
            ss = VectorSpace{Float32}(vcat(-ones(Float32, n), zeros(Float32, n)), vcat(ones(Float32, n), ones(Float32, n)))
        else
            ss = VectorSpace{Float32}(vcat(Float32.(env_ss_lows), -ones(Float32, n), zeros(Float32, n)), vcat(Float32.(env_ss_highs), ones(Float32, n), ones(Float32, n)))
        end
        nactions = n
        if S == Int
            nstates = m
        else
            nstates = nothing
        end
        if isnothing(nstates)
            model = LearnedTabularModel(env; nactions=nactions, abstraction_radius=abstraction_radius, abstraction_cluster_size=abstraction_cluster_size)
            prealloc_Q = nothing
            prealloc_V = nothing
            prealloc_T = nothing
            prealloc_R = nothing
        else
            model = LearnedTabularModel(env; nstates=nstates, nactions=nactions, abstraction_radius=abstraction_radius, abstraction_cluster_size=abstraction_cluster_size)
            prealloc_Q = zeros(Float64, nactions, nstates)
            prealloc_V = zeros(Float64, nstates)
            prealloc_T = zeros(Float64, nstates, nactions, nstates)
            prealloc_R = zeros(Float64, nstates, nactions, nstates)
        end
        new{S}(env, 0, drop_observation, task_horizon, abstraction_radius, abstraction_cluster_size, VI_EP, Q_DENOM, ss, state(env), false, Dict{S, Vector{Float32}}(), Dict{S, Vector{Int}}(), model, Dict{S, Float32}(), drop_observation ? zeros(Float32, 2n) : zeros(Float32, m + 2n), prealloc_Q, prealloc_V, prealloc_T, prealloc_R)
    end
end

function factory_reset!(env::ValueEstimateAugmentedMDPER{S}) where {S}
    # println("here")
    factory_reset!(env.env)
    empty!(env.q)
    empty!(env.v)
    empty!(env.counts)
    nactions = length(action_space(env.env))
    if S == Int
        nstates = length(state_space(env.env))
    else
        nstates = nothing
    end
    if isnothing(nstates)
        env.model = LearnedTabularModel(env.env; nactions=nactions, abstraction_radius=env.abstraction_radius, abstraction_cluster_size=env.abstraction_cluster_size)
    else
        env.model = LearnedTabularModel(env.env; nstates=nstates, nactions=nactions, abstraction_radius=env.abstraction_radius, abstraction_cluster_size=env.abstraction_cluster_size)
    end
    nothing
end

action_space(env::ValueEstimateAugmentedMDPER) = action_space(env.env)
state_space(env::ValueEstimateAugmentedMDPER) = env.ss
action_meaning(env::ValueEstimateAugmentedMDPER, a::Int) = action_meaning(env.env, a)

state(env::ValueEstimateAugmentedMDPER) = env.state
action(env::ValueEstimateAugmentedMDPER) = action(env.env)
reward(env::ValueEstimateAugmentedMDPER) = reward(env.env)

function reset!(env::ValueEstimateAugmentedMDPER{S}; rng::AbstractRNG=Random.GLOBAL_RNG) where {S}
    # println("in reset")
    reset!(env.env; rng=rng)
    if S == Int
        env.o = state(env.env)
    else
        copy!(env.o, state(env.env))
    end
    _o = copy(env.o)
    env.is_start = true

    # in the following code, if _o is not in the dictionary, then we add it and initialize env.v, env.q, and env.counts.
    if !haskey(env.q, _o)
        env.q[_o] = zeros(Float32, length(action_space(env.env)))
        env.counts[_o] = zeros(Int, length(action_space(env.env)))
        env.v[_o] = 0
    end
    
    o = S == Int ? to_onehot(env.o, length(state_space(env.env)), Float32) : convert(Vector{Float32}, env.o)
    env.state .= env.drop_observation ? vcat(env.q[env.o] / env.Q_DENOM, env.counts[env.o] / 100)  : vcat(o, env.q[env.o] / env.Q_DENOM, env.counts[env.o] / 100)
    nothing
end


function step!(env::ValueEstimateAugmentedMDPER{S}, a::Int; rng=AbstractRNG=Random.GLOBAL_RNG) where {S}
    # println("in step")
    _o::S = copy(env.o)

    step!(env.env, a; rng=rng)

    r::Float64, o′::S, d::Bool, γ::Float64 = reward(env.env), state(env.env), in_absorbing_state(env.env), 0.99
    _o′::S = copy(o′)

    if !haskey(env.q, _o′)
        env.q[_o′] = zeros(Float32, length(action_space(env.env)))
        env.counts[_o′] = zeros(Int, length(action_space(env.env)))
        env.v[_o′] = 0
    end

    # update env.model:
    update_model!(env.model, env.is_start, _o, a, r, _o′, d)
    if isnothing(env.prealloc_Q)
        env.prealloc_Q = zeros(Float64, env.model.nactions, env.model.nstates)
        env.prealloc_V = zeros(Float64, env.model.nstates)
        env.prealloc_T = zeros(Float64, env.model.nstates, env.model.nactions, env.model.nstates)
        env.prealloc_R = zeros(Float64, env.model.nstates, env.model.nactions, env.model.nstates)
        value_iteration!(env.model, 0.99, env.task_horizon; q=env.prealloc_Q, v=env.prealloc_V, T=env.prealloc_T, R=env.prealloc_R, ϵ=env.VI_EP)
    else
        if size(env.prealloc_Q) == (env.model.nactions, env.model.nstates)
            # println("yeah! ", size(env.prealloc_Q))
            value_iteration!(env.model, 0.99, env.task_horizon; q=env.prealloc_Q, v=env.prealloc_V, T=env.prealloc_T, R=env.prealloc_R, ϵ=env.VI_EP)
        else
            # println("damn! ", (env.model.nactions, env.model.nstates))
            # @time begin
                env.prealloc_Q = zeros(Float64, env.model.nactions, env.model.nstates)
                env.prealloc_V = zeros(Float64, env.model.nstates)
                env.prealloc_T = zeros(Float64, env.model.nstates, env.model.nactions, env.model.nstates)
                env.prealloc_R = zeros(Float64, env.model.nstates, env.model.nactions, env.model.nstates)
                value_iteration!(env.model, 0.99, env.task_horizon; q=env.prealloc_Q, v=env.prealloc_V, T=env.prealloc_T, R=env.prealloc_R, ϵ=env.VI_EP)
            # end
        end
    end
    # println(size(env.prealloc_Q), (env.model.nactions, env.model.nstates))
    V, Q = env.prealloc_V, env.prealloc_Q
    
    # update env.q and env.count for _o. Remember that Julia is column-major.
    o_id = get_state_id(env.model, _o)
    # println("state id of from state is ", o_id)
    env.q[_o] .= Q[:, o_id]
    env.counts[_o][a] += 1
    env.is_start = false

    # update env.v for all states in the model:
    for id in 1:env.model.nstates
        all_o_with_this_id = get_all_states_with_id(env.model, id)
        for o in all_o_with_this_id
            env.v[o] = V[id]
        end
    end

    # ------------------------------------------------------------------

    if S == Int
        env.o = o′
    else
        copy!(env.o, o′)
    end
    o = S == Int ? to_onehot(env.o, length(state_space(env.env)), Float32) : convert(Vector{Float32}, env.o)
    env.state .= env.drop_observation ? vcat(env.q[env.o] / env.Q_DENOM, env.counts[env.o] / 100) : vcat(o, env.q[env.o] / env.Q_DENOM, env.counts[env.o] / 100)
    nothing
end

in_absorbing_state(env::ValueEstimateAugmentedMDPER) = in_absorbing_state(env.env)

function visualize(env::ValueEstimateAugmentedMDPER, args...; kwargs...)
    visualize(env.env, args...; value_fn=env.v, kwargs...)
end

function value_iteration!(mdp::AbstractMDP{Int, Int}, γ::Float64, horizon::Int; ϵ::Float64=0.01, q::Matrix{Float64}, v::Vector{Float64}, T::Array{Float64, 3}, R::Array{Float64, 3})::Nothing
    nstates::Int = length(state_space(mdp))
    nactions::Int = length(action_space(mdp))

    @assert size(q) == (nactions, nstates)
    @assert length(v) == nstates
    @assert size(T) == (nstates, nactions, nstates)
    @assert size(R) == (nstates, nactions, nstates)
    for s::Int in 1:nstates
        for a::Int in 1:nactions
            for s′::Int in 1:nstates
                @inbounds T[s′, a, s] = transition_probability(mdp, s, a, s′)
                @inbounds R[s′, a, s] = reward(mdp, s, a, s′)
            end
        end
    end

    i::Int = 0
    while i < horizon
        δ =ValueIteration.bellman_backup_synchronous!(q, v, R, T, γ)
        for s::Int in 1:nstates
            @inbounds v[s] = @views maximum(q[:, s])
        end
        i += 1
        δ < ϵ && break
    end

    return nothing
end