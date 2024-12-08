using MDPs
import MDPs
using DataStructures
using StatsBase

"""
    LearnedTabularModel{S, A} <: AbstractMDP{Int, Int}  # S and A are the types of states and actions in the environment.

    This is a tabular MDP model that is learned from experience. If states or actions are continuous, they can be discretized. The model is updated incrementally as new experience is observed.

    Fields:

    state_counts::Dict{S, Int} = Dict{S, Int}() # counts of each state encountered so far
    action_counts::Dict{A, Int} = Dict{A, Int}() # counts of each action encountered so far
    actions_support::Dict{Int, Set{Int}} = Dict{Int, Set{Int}}() # set of actions that can be taken in each state
    T_counts::Dict{Tuple{Int, Int}, Int} = Dict{Tuple{Int, Int}, Int}() # counts of transitions encountered so far given a state and action.
    T_support::Dict{Tuple{Int, Int}, Set{Int}} = Dict{Tuple{Int, Int}, Set{Int}}() # set of next states that can be reached given a state and action
    T_counts_nextstate::Dict{Tuple{Int, Int}, Dict{Int, Int}} = Dict{Tuple{Int, Int}, Dict{Int, Int}}() # counts of each next state encountered so far given a state and action.
    T::Dict{Tuple{Int, Int}, Dict{Int, Float64}} = Dict{Tuple{Int, Int}, Dict{Int, Float64}}() # transition probabilities of each next state given a state and action
    experience_tuple_counts::Dict{Tuple{Int, Int, Int}, Int} = Dict{Tuple{Int, Int, Int}, Int}() # counts of each experience tuple encountered so far
    R::Dict{Tuple{Int, Int, Int}, Float64} = Dict{Tuple{Int, Int, Int}, Float64}() # average reward for each experience tuple
    r_max::Float64 = 0 # maximum absolute reward encountered so far
    num_trajs::Int = 0 # number of trajectories seen so far
    d₀_support::Set{Int} = Set{Int}() # set of initial states encountered so far
    d₀_counts::Dict{Int, Int} = Dict{Int, Int}() # counts of each initial state encountered so far
    d₀::Dict{Int, Float64} = Dict{Int, Float64}() # initial state probabilities
    absorbing_states::Set{Int} = Set{Int}() # set of absorbing states encountered so far
    nstates::Int = 0 # number of states inferred to be in the MDP so far. Equal to the highest state id encountered so far.
    nactions::Int = 0 # number of actions inferred to be in the MDP so far. Equal to the highest action id encountered so far.
    state_id::Dict{S, Int} = Dict{S, Int}() # map from state to its id
    action_id::Dict{A, Int} = Dict{A, Int}() # map from action to its id
    state_id_reverse_map::Dict{Int, Set{S}} = Dict{Int, Set{S}}() # map from state id to the set of states that have been assigned this id
    action_id_reverse_map::Dict{Int, Set{A}} = Dict{Int, Set{A}}() # map from action id to the set of actions that have been assigned this id
    get_action_id_fn = nothing # maps an action to an action id. Function args: (LearnedMDP, a::A). By default: The action id is the same as the action itself if A is an integer type. If A is not an integer type, then a new action_id is assigned to each new action encountered.
    get_state_id_fn = nothing # maps a state to a state id. Function args: (LearnedMDP, s::S). By default: The state id is the same as the state itself if S is an integer type. If S is not an integer type, then a new state_id is assigned to each new state encountered.
    abstraction_radius = 0   # modifies the default behavior of assigning state IDs when get_state_id_fn is nothing and states are continuous. When a new concrete state is encountered, assign its abstract state ID to that of an old state within the abstraction_radius (by euclidean distance), unless the old state is already part of a full cluster (determined by `abstraction_cluster_size`). If there are multiple old states that satisfy the criteria, cluster the new state with the closet old state. If there are no old states that satisfy the criteria, then assign the new state to a new abstract state ID.
    abstraction_cluster_size = 1 
    default_reward = 0.0  # Unobserved actions in a given state will be assumed to have a NOOP effect while yielding reward = `default_reward`.
"""
Base.@kwdef mutable struct LearnedTabularModel{S, A} <: AbstractMDP{Int, Int}
    laplace_smoothing::Float64 = 0.1
    state_counts::Dict{S, Int} = Dict{S, Int}()
    action_counts::Dict{A, Int} = Dict{A, Int}()
    actions_support::Dict{Int, Set{Int}} = Dict{Int, Set{Int}}()
    T_counts::Dict{Tuple{Int, Int}, Int} = Dict{Tuple{Int, Int}, Int}()
    T_support::Dict{Tuple{Int, Int}, Set{Int}} = Dict{Tuple{Int, Int}, Set{Int}}()
    T_counts_nextstate::Dict{Tuple{Int, Int}, Dict{Int, Int}} = Dict{Tuple{Int, Int}, Dict{Int, Int}}()
    T::Dict{Tuple{Int, Int}, Dict{Int, Float64}} = Dict{Tuple{Int, Int}, Dict{Int, Float64}}()
    experience_tuple_counts::Dict{Tuple{Int, Int, Int}, Int} = Dict{Tuple{Int, Int, Int}, Int}()
    R::Dict{Tuple{Int, Int, Int}, Float64} = Dict{Tuple{Int, Int, Int}, Float64}()
    R_squared::Dict{Tuple{Int, Int, Int}, Float64} = Dict{Tuple{Int, Int, Int}, Float64}()
    r_max::Float64 = 0
    num_trajs::Int = 0
    d₀_support::Set{Int} = Set{Int}()
    d₀_counts::Dict{Int, Int} = Dict{Int, Int}()
    d₀::Dict{Int, Float64} = Dict{Int, Float64}()
    absorbing_states::Set{Int} = Set{Int}()
    nstates::Int = 0
    nactions::Int = 0
    state_id::Dict{S, Int} = Dict{S, Int}()
    action_id::Dict{A, Int} = Dict{A, Int}()
    state_id_reverse_map::Dict{Int, Set{S}} = Dict{Int, Set{S}}()
    action_id_reverse_map::Dict{Int, Set{A}} = Dict{Int, Set{A}}()
    get_action_id_fn = nothing # maps an action to an action id. Function args: (LearnedMDP, a::A)
    get_state_id_fn = nothing # maps a state to a state id. Function args: (LearnedMDP, s::S)
    abstraction_radius::Float64 = 0
    abstraction_cluster_size::Int = 1
    default_reward = 0.0 # Unobserved actions in a given state will be assumed to have a NOOP effect while yielding reward = `default_reward`.

    # To use this MDP as an RL env, need to have the following fields:
    state::Int = 1 
    action::Int = 1 # action taken in the previous state
    reward::Float64 = 0.0  # reward received in the previous state
end

LearnedTabularModel(env::AbstractMDP{S, A}; kwargs...) where {S, A} = LearnedTabularModel{S, A}(; kwargs...)


"""
    get_action_id(lmdp::LearnedTabularModel{S, A}, a::A; increment_action_count=false)::Int

Returns the id of the action `a` in `lmdp`.
If `increment_action_count` is true, then the count of the action is incremented by 1.
"""
function get_action_id(lmdp::LearnedTabularModel{S, A}, a::A; increment_action_count=false)::Int where {S, A}
    if haskey(lmdp.action_id, a)
        id = lmdp.action_id[a]
        if increment_action_count
            lmdp.action_counts[a] += 1
        end
    else
        if isnothing(lmdp.get_action_id_fn)
            if isa(a, Integer)
                id = a
            else
                id = lmdp.nactions + 1
            end
        else
            id = lmdp.get_action_id_fn(lmdp, a)
        end

        lmdp.action_id[a] = id
        lmdp.action_counts[a] = 1
        if !haskey(lmdp.action_id_reverse_map, id)
            lmdp.action_id_reverse_map[id] = Set{A}()
        end
        push!(lmdp.action_id_reverse_map[id], a)
        lmdp.nactions = max(lmdp.nactions, id)
    end
    return id
end


"""
    get_state_id(lmdp::LearnedTabularModel{S, A}, s::S; increment_state_count=false)::Int

Returns the id of the state `s` in `lmdp`.
If `increment_state_count` is true, then the count of the state is incremented by 1.
"""
function get_state_id(lmdp::LearnedTabularModel{S, A}, s::S; increment_state_count=false)::Int where {S, A}
    if haskey(lmdp.state_id, s)
        id = lmdp.state_id[s]
        if increment_state_count
            lmdp.state_counts[s] += 1
        end
    else
        if isnothing(lmdp.get_state_id_fn)
            if isa(s, Integer)
                id = s
            else
                # id = lmdp.nstates + 1
                id = assign_abstract_state_id(lmdp, s)
                # println("$s assigned to $id")
            end
        else
            id = lmdp.get_state_id_fn(lmdp, s)
        end
        lmdp.state_id[s] = id
        lmdp.state_counts[s] = 1
        if !haskey(lmdp.state_id_reverse_map, id)
            lmdp.state_id_reverse_map[id] = Set{S}()
        end
        push!(lmdp.state_id_reverse_map[id], s)
        lmdp.nstates = max(lmdp.nstates, id)
    end
    return id
end

"""
    get_action_from_action_id(lmdp::LearnedTabularModel{S, A}, id::Int)::A where {S, A}
    
    Returns an action `a` such that `get_action_id(lmdp, a) == id`. If there are multiple actions with the same id, then one of them is sampled with probability proportional to the number of times it has been seen in the data.
"""
function get_action_from_action_id(lmdp::LearnedTabularModel{S, A}, id::Int)::A where {S, A}
    a_possibilities = collect(get(lmdp.action_id_reverse_map, id, Set{A}()))
    @assert length(a_possibilities) > 0 "There are no actions mapped to id $id"
    a_probability_weights = [lmdp.action_counts[a] for a in a_possibilities]
    a = sample(a_possibilities, ProbabilityWeights(a_probability_weights))
    return a
end

"""
    get_state_from_state_id(lmdp::LearnedTabularModel{S, A}, id::Int)::S where {S, A}

Returns a state `s` such that `get_state_id(lmdp, s) == id`. If there are multiple states with the same id, then one of them is sampled with probability proportional to the number of times it has been seen in the data.

"""
function get_state_from_state_id(lmdp::LearnedTabularModel{S, A}, id::Int)::S where {S, A}
    s_possibilities = collect(get(lmdp.state_id_reverse_map, id, Set{S}()))
    @assert length(s_possibilities) > 0 "There are no states mapped to this id $id"
    s_probability_weights = [lmdp.state_counts[s] for s in s_possibilities]
    s = sample(s_possibilities, ProbabilityWeights(s_probability_weights))
    return s
end


"""
    get_all_states_with_id(lmdp::LearnedTabularModel{S, A}, id::Int)::Set{S} where {S, A}

Returns a set of states `s` such that `get_state_id(lmdp, s) == id`. Returns an empty set if there are no states with this id.
"""
function get_all_states_with_id(lmdp::LearnedTabularModel{S, A}, id::Int)::Set{S} where {S, A}
    return get(lmdp.state_id_reverse_map, id, Set{S}())
end


"""
    get_all_actions_with_id(lmdp::LearnedTabularModel{S, A}, id::Int)::Set{A} where {S, A}

Returns a set of actions `a` such that `get_action_id(lmdp, a) == id`. Returns an empty set if there are no actions with this id.
"""
function get_all_actions_with_id(lmdp::LearnedTabularModel{S, A}, id::Int)::Set{A} where {S, A}
    return get(lmdp.action_id_reverse_map, id, Set{A}())
end

"""
    update_model!(lmdp::LearnedTabularModel{S, A}, is_start_state::Bool, s::S, a::A, r::Float64, s′::S, is_absorbing_state::Bool)::Nothing where {S, A}

Updates the learned MDP with the transition (s, a, r, s′).
If `is_start_state` is true, then the state `s` is assumed to be the start state.
If `is_absorbing_state` is true, then the state `s′` is assumed to be an absorbing state.
"""
function update_model!(lmdp::LearnedTabularModel{S, A}, is_start_state::Bool, s::S, a::A, r::Float64, s′::S, is_absorbing_state::Bool)::Nothing where {S, A}

    _s::Int = get_state_id(lmdp, s; increment_state_count=true)
    _a::Int = get_action_id(lmdp, a; increment_action_count=true)
    _s′::Int = get_state_id(lmdp, s′; increment_state_count=is_absorbing_state)

    # println((s ,a, s′), " -> ", (_s, _a, _s′))

    # update legal actions:    
    if !haskey(lmdp.actions_support, _s)
        lmdp.actions_support[_s] = Set{Int}()
    end
    push!(lmdp.actions_support[_s], _a)

    # update transition function:
    if !haskey(lmdp.T_counts_nextstate, (_s, _a))
        lmdp.T_counts_nextstate[(_s, _a)] = Dict{Int, Int}()
    end
    lmdp.T_counts[(_s, _a)] = get(lmdp.T_counts, (_s, _a), 0) + 1
    lmdp.T_counts_nextstate[(_s, _a)][_s′] = get(lmdp.T_counts_nextstate[(_s, _a)], _s′, 0) + 1

    # update reward function:
    lmdp.experience_tuple_counts[(_s, _a, _s′)] = get(lmdp.experience_tuple_counts, (_s, _a, _s′), 0) + 1  # new count
    n = lmdp.experience_tuple_counts[(_s, _a, _s′)]
    lmdp.R[(_s, _a, _s′)] = ((n - 1) * get(lmdp.R, (_s, _a, _s′), 0)  + r) / n
    lmdp.R_squared[(_s, _a, _s′)] = ((n - 1) * get(lmdp.R_squared, (_s, _a, _s′), 0)  + r^2) / n
    lmdp.r_max = max(lmdp.r_max, abs(r))

    # update start state distribution
    if is_start_state
        lmdp.num_trajs += 1
        push!(lmdp.d₀_support, _s)
        lmdp.d₀_counts[_s] = get(lmdp.d₀_counts, _s, 0) + 1
        for _s₀ in lmdp.d₀_support
            lmdp.d₀[_s₀] = lmdp.d₀_counts[_s₀] / lmdp.num_trajs
        end
    end

    # update set of absorbing states
    if is_absorbing_state
        push!(lmdp.absorbing_states, _s′)
    end
    
    nothing
end


MDPs.state_space(lmdp::LearnedTabularModel) = IntegerSpace(lmdp.nstates)

MDPs.action_space(lmdp::LearnedTabularModel) = IntegerSpace(lmdp.nactions)

function MDPs.start_state_support(lmdp::LearnedTabularModel)::Set{Int}
    return lmdp.d₀_support
end

function MDPs.start_state_probability(lmdp::LearnedTabularModel, s::Int)::Float64
    get(lmdp.d₀, s, 0)
end

function MDPs.transition_probability(lmdp::LearnedTabularModel, s::Int, a::Int, s′::Int)::Float64
    if s ∈ lmdp.absorbing_states
        return s′ == s ? 1.0 : 0.0
    else
        Nsa = get(lmdp.T_counts, (s, a), 0)
        if Nsa == 0 # have never seen this s-a pair. return uniform distribution over all state outcomes.
            return 1.0 / lmdp.nstates
        else
            Nsas′ = get(lmdp.T_counts_nextstate[(s, a)], s′, 0)
            numer = Nsas′ + lmdp.laplace_smoothing  # Laplace smoothing
            denom = Nsa + lmdp.laplace_smoothing * lmdp.nstates  # Laplace smoothing
            return numer / denom
        end
    end
end

function standard_error_transition(lmdp::LearnedTabularModel, s::Int, a::Int, s′::Int)::Float64
    Nsa = get(lmdp.T_counts, (s, a), 0)
    if Nsa == 0 # we havn't tried this action. return standard deviation of uniform distribution between 0 and 1. i.e., assume anything could happen.
        return sqrt(1/12)
        # return 0.0
    else
        p = MDPs.transition_probability(lmdp, s, a, s′)
        n = Nsa + lmdp.laplace_smoothing * lmdp.nstates  # Laplace smoothing
        return sqrt(p * (1 - p) / n)  # standard error of a proportion. Assume collapse of uncertainty (caveat: Laplace smoothing) even with a single observation.
    end
end

function MDPs.reward(lmdp::LearnedTabularModel, s::Int, a::Int, s′::Int)::Float64
    if s ∈ lmdp.absorbing_states
        return 0
    else
        return get(lmdp.R, (s, a, s′), lmdp.default_reward)
    end
end

function reward_square(lmdp::LearnedTabularModel, s::Int, a::Int, s′::Int)::Float64
    if s ∈ lmdp.absorbing_states
        return 0
    else
        return get(lmdp.R_squared, (s, a, s′), lmdp.default_reward^2)
    end
end

function standard_error_reward(lmdp::LearnedTabularModel, s::Int, a::Int, s′::Int)::Float64
    n = get(lmdp.experience_tuple_counts, (s, a, s′), 1)
    if n <= 1 # return a prior
        return 1.0
    else
        r, r2 = MDPs.reward(lmdp, s, a, s′), reward_square(lmdp, s, a, s′)
        std = sqrt(max(r2 - r^2, 0))
        ste = std / sqrt(n - 1)  # Bessel's corrections
        return ste
    end
end

function MDPs.is_absorbing(lmdp::LearnedTabularModel, s::Int)
    return s ∈ lmdp.absorbing_states
end


function MDPs.truncated(lmdp::LearnedTabularModel)
    # When using this model as an RL simulator, need to truncate the interaction if there is no known transition from the current state
    return length(lmdp.actions_support[lmdp.state]) == 0
end


# -----------------------------------------------------------------------

"""
    TabularModelLearnerHook

An hook that updates a `LearnedTabularModel` after each step is taken.
"""
Base.@kwdef mutable struct TabularModelLearnerHook{S, A} <: AbstractHook
    const model::LearnedTabularModel{S, A} = LearnedTabularModel{S, A}()
    isstart::Bool = true
    s::Union{Nothing, S} = nothing
end
TabularModelLearnerHook(model::LearnedTabularModel{S, A}) where {S, A} = TabularModelLearnerHook{S, A}(model=model)

function MDPs.preepisode(mlh::TabularModelLearnerHook{S, A}; env::AbstractMDP{S, A}, kwargs...) where {S, A}
    mlh.isstart = true
end

function MDPs.prestep(mlh::TabularModelLearnerHook{S, A}; env::AbstractMDP{S, A}, kwargs...) where {S, A}
    mlh.s = state(env)
end

function MDPs.poststep(mlh::TabularModelLearnerHook{S, A}; env::AbstractMDP{S, A}, kwargs...) where {S, A}
    update_model!(mlh.model, mlh.isstart, mlh.s, action(env), reward(env), state(env), in_absorbing_state(env))
    mlh.isstart = false
end



# Import required modules
using LinearAlgebra: norm

function assign_abstract_state_id(lmdp::LearnedTabularModel{S, A}, concrete_state::S)::Int where {S <: AbstractVector{<:Real}, A}
    abstract_state_dict = lmdp.state_id
    cluster_size(id) = length(get_all_states_with_id(lmdp, id))
    abstract_state_counter = lmdp.nstates + 1
    bucketing_radius = lmdp.abstraction_radius
    max_bucket_size = lmdp.abstraction_cluster_size

    # Check if the concrete state has been encountered before
    if haskey(abstract_state_dict, concrete_state)
        println(abstract_state_dict)
        return abstract_state_dict[concrete_state]
    end

    # Find the closest old state within the bucketing radius that is not part of a full cluster
    closest_abstract_state_id = -1
    if bucketing_radius > 0 && max_bucket_size > 1
        min_distance = Inf
        for (old_state, abstract_state_id) in abstract_state_dict
            if cluster_size(abstract_state_id) < max_bucket_size
                distance = norm(concrete_state - old_state)
                if distance < bucketing_radius && distance < min_distance
                    min_distance = distance
                    closest_abstract_state_id = abstract_state_id
                end
            end
        end
    end

    # If there are no old states that satisfy the criteria, assign the new state to a new abstract state ID
    if closest_abstract_state_id == -1
        closest_abstract_state_id = abstract_state_counter
    end
    
    return closest_abstract_state_id
end
