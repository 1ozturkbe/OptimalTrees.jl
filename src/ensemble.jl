
""" Contains default TreeEnsemble parameters, and modifies them with kwargs. """
function TreeEnsemble_defaults(kwargs...)
    d = Dict(:max_depth => 2,
        :cp => 1e-4,
        :hypertol => 0.005, # hyperplane separation tolerance
        :minbucket => 0.02, 
        :regression => false)
    set_params!(d; kwargs...)
    return d
end

mutable struct TreeEnsemble
    trees::Array
    weights::Array
    params::Dict
    classes::Union{Nothing, Array{Any}}
    solver

    function TreeEnsemble(solver; kwargs...)
        te = new([],
                 [],
                 TreeEnsemble_defaults(),
                 nothing,
                 solver)
        set_params!(te.params; kwargs...)
        return te
    end
end

set_param!(te::TreeEnsemble, s::Symbol, v::Any) = set_param!(te.params, s, v)
set_params!(te::TreeEnsemble; kwargs...) = set_params!(te.params; kwargs...)
get_param(te::TreeEnsemble, s::Symbol) = get_param(te.params, s)

""" Plants a set number of MIOTrees in a TreeEnsemble. """
function plant_trees(te::TreeEnsemble, n_trees::Int)
    solver = te.solver
    for i = 1:n_trees
        push!(te.trees, MIOTree(solver; te.params...))
    end
end

function fit!(te::TreeEnsemble, method::String, X, Y)
    data = split_data(X, Y, bins = length(te.trees))
    if !get_param(te, :regression)
        te.classes = sort(unique(Y)) # Make sure that all trees have the same classes. 
        for mt in te.trees
            mt.classes = te.classes
        end
    end
    @showprogress 1 "Training ensemble of $(length(te.trees)) trees. " for i = 1:length(te.trees)
        fit!(te.trees[i], method, data[i][1], data[i][2])
    end
    return
end

""" Computes the optimal weights for a regressing TreeEnsemble. """
function weigh_trees(te, X, Y)
    get_param(te, :regression) || throw(ErrorException("Can only weight TreeEnsembles for regression."))
    m = JuMP.Model(te.solver)
    @variable(m, w[1:length(te.trees)])
    @constraint(m, sum(w) == 1)
    @variable(m, preds[1:length(Y), 1:length(te.trees)])
    evals = hcat([predict(mt, X) for mt in te.trees]...)
    @constraint(m, preds .== evals * w)
    @objective(m, Min, 1/length(Y)*sum((Y .- preds).^2)) # Minimize squared error
    optimize!(m)
    te.weights = getvalue.(w)
    return
end

function predict(te::TreeEnsemble, X)
    evals = hcat([predict(mt, X) for mt in te.trees]...)
    if get_param(te, :regression)
        if isnothing(te.weights)
            throw(ErrorException("TreeEnsemble must be weighted before prediction. Please use the weigh_trees function."))
        else
            return evals * te.weights
        end
    else
        return [mode(evals[i,:])[1] for i = 1:size(X, 1)] # TODO: find a better way to tie-break?
    end
end

function score(te::TreeEnsemble, X, Y)
    preds = predict(te, X)
    if get_param(te, :regression)
        return 1 - sum((preds .- Y).^2) / sum((preds .- sum(Y)/length(Y)*ones(length(Y))).^2)
    else
        return sum(preds .== Y)/length(Y)
    end
end