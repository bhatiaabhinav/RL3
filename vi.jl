function ste_value_iteration!(
    mdp::AbstractMDP{Int, Int},
    γ::Float64,
    horizon::Int;
    ϵ::Float64 = 0.01,
    q::Matrix{Float64},
    v::Vector{Float64},
    T::Array{Float64, 3},
    R::Array{Float64, 3},
    E_T::Array{Float64, 3},
    E_R::Array{Float64, 3},
    E_q::Matrix{Float64},
    E_v::Vector{Float64},
    omit_standard_errors::Bool = false
)::Nothing

    if omit_standard_errors
        return value_iteration!(mdp, γ, horizon; ϵ = ϵ, q = q, v = v, T = T, R = R)
    end
    nstates::Int = length(state_space(mdp))
    nactions::Int = length(action_space(mdp))

    @assert size(q) == (nactions, nstates)
    @assert length(v) == nstates
    @assert size(T) == (nstates, nactions, nstates)
    @assert size(R) == (nstates, nactions, nstates)
    @assert size(E_T) == (nstates, nactions, nstates)
    @assert size(E_R) == (nstates, nactions, nstates)
    @assert size(E_q) == (nactions, nstates)
    @assert length(E_v) == nstates

    # Initialize T, R, E_T, and E_R (assuming E_T and E_R are provided externally based on data)
    Threads.@threads for s::Int in 1:nstates
        for a::Int in 1:nactions
            for s′::Int in 1:nstates
                @inbounds T[s′, a, s] = transition_probability(mdp, s, a, s′)
                @inbounds R[s′, a, s] = reward(mdp, s, a, s′)
                @inbounds E_T[s′, a, s] = standard_error_transition(mdp, s, a, s′)
                @inbounds E_R[s′, a, s] = standard_error_reward(mdp, s, a, s′)
            end
        end
    end

    i::Int = 0
    while i < horizon
        δ = ste_bellman_backup_synchronous!(q, v, R, T, γ, E_q, E_R, E_T, E_v)
        for s::Int in 1:nstates
            a = @views argmax(q[:, s])
            v[s] = q[a, s]
            E_v[s] = E_q[a, s]
        end
        i += 1
        δ < ϵ && break
    end

    return nothing
end

function ste_bellman_backup_synchronous!(
    q::Matrix{Float64},
    v::Vector{Float64},
    R::Array{Float64, 3},
    T::Array{Float64, 3},
    γ::Float64,
    E_q::Matrix{Float64},
    E_R::Array{Float64, 3},
    E_T::Array{Float64, 3},
    E_v::Vector{Float64}
)::Float64
    nactions::Int, nstates::Int = size(q)
    δ::Float64 = 0.0

    for s::Int in 1:nstates
        for a::Int in 1:nactions
            q_new::Float64 = 0.0
            Var_q_new::Float64 = 0.0

            @inbounds for s′::Int in 1:nstates
                # Retrieve estimates and standard errors
                T_sa_s′::Float64 = T[s′, a, s]
                R_sa_s′::Float64 = R[s′, a, s]
                v_s′::Float64 = v[s′]
                E_T_sa_s′::Float64 = E_T[s′, a, s]
                E_R_sa_s′::Float64 = E_R[s′, a, s]
                E_v_s′::Float64 = E_v[s′]

                # Compute the expected value
                temp::Float64 = T_sa_s′ * (R_sa_s′ + γ * v_s′)
                q_new += temp

                # Variance propagation
                Var_Rv::Float64 = E_R_sa_s′^2 + γ^2 * E_v_s′^2
                Var_T::Float64 = E_T_sa_s′^2
                Var_temp::Float64 = T_sa_s′^2 * Var_Rv + (R_sa_s′ + γ * v_s′)^2 * Var_T + Var_Rv * Var_T
                Var_q_new += Var_temp
            end

            # Update δ
            δₐₛ::Float64 = abs(q_new - q[a, s])
            if δₐₛ > δ
                δ = δₐₛ
            end

            # Update Q-value and its standard error
            q[a, s] = q_new
            E_q[a, s] = sqrt(max(Var_q_new, 0.0))
        end
    end
    return δ
end



function value_iteration!(mdp::AbstractMDP{Int, Int}, γ::Float64, horizon::Int; ϵ::Float64=0.01, q::Matrix{Float64}, v::Vector{Float64}, T::Array{Float64, 3}, R::Array{Float64, 3})::Nothing
    nstates::Int = length(state_space(mdp))
    nactions::Int = length(action_space(mdp))

    @assert size(q) == (nactions, nstates)
    @assert length(v) == nstates
    @assert size(T) == (nstates, nactions, nstates)
    @assert size(R) == (nstates, nactions, nstates)
    Threads.@threads for s::Int in 1:nstates
        for a::Int in 1:nactions
            for s′::Int in 1:nstates
                @inbounds T[s′, a, s] = transition_probability(mdp, s, a, s′)
                @inbounds R[s′, a, s] = reward(mdp, s, a, s′)
            end
        end
    end

    i::Int = 0
    while i < horizon
        δ =bellman_backup_synchronous!(q, v, R, T, γ) 
        for s::Int in 1:nstates
            @inbounds v[s] = @views maximum(q[:, s])
        end
        i += 1
        δ < ϵ && break
    end

    return nothing
end

function bellman_backup_synchronous!(q::Matrix{Float64}, v::Vector{Float64}, R::Array{Float64, 3}, T::Array{Float64, 3}, γ::Float64)::Float64
    nactions::Int, nstates::Int = size(q)
    δ::Float64 = 0
    for s::Int in 1:nstates
        @inbounds for a::Int in 1:nactions
            qᵢ₊₁::Float64 = 0
            @simd for s′ in 1:nstates
                qᵢ₊₁ += T[s′, a, s]*(R[s′, a, s] + γ * v[s′])
            end
            δₐₛ::Float64 = abs(qᵢ₊₁ - q[a, s])
            if δₐₛ > δ
                δ = δₐₛ
            end
            q[a, s] = qᵢ₊₁
        end
    end
    return δ
end