using MDPs
using BSON

struct StatsPrintHook <: AbstractHook
    stats_getter # some callable returning a stats dictionary
    n::Int
end

function MDPs.postepisode(sph::StatsPrintHook; env, steps, returns, lengths, kwargs...)
    episodes = length(returns)
    if episodes % sph.n == 0
        min_recs = 100
        R̄ =  length(returns) < min_recs ? mean(returns) : mean(returns[end-min_recs+1:end])
        R = returns[end]
        L = lengths[end]
        stats::Dict{Symbol, Any} = sph.stats_getter()
        @info "stats" steps episodes R̄ R L stats...
    end
end



struct GCHook <: AbstractHook
end

function MDPs.postepisode(::GCHook; kwargs...)
    GC.gc()
end



struct ModelsSaveHook <: AbstractHook
    models
    dirname::String
    interval::Int
end
function MDPs.postepisode(msh::ModelsSaveHook; returns, steps, kwargs...)
    if length(returns) % msh.interval == 0
        mkpath("models/$(msh.dirname)")
        models = msh.models
        BSON.@save "models/$(msh.dirname)/ep-$(length(returns))-steps-$(steps).bson" models
    end
end
function MDPs.postexperiment(msh::ModelsSaveHook; returns, steps, kwargs...)
    mkpath("models/")
    models = msh.models
    BSON.@save "models/$(msh.dirname)/ep-$(length(returns))-steps-$(steps).bson" models
end

function loadmodels(filename)
    BSON.@load filename models
    return models
end

