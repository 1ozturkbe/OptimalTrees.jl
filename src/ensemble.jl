
""" Contains default TreeEnsemble parameters, and modifies them with kwargs. """
function TreeEnsemble_defaults(kwargs...)
    d = Dict(:max_depth => 2,
        :cp => 1e-4,
        :hypertol => 0.005, # hyperplane separation tolerance
        :minbucket => 0.02, 
        :regression => false)
    if !isempty(kwargs)
        for (key, value) in kwargs
            set_param(d, key, value)
        end
    end
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
        for (key, val) in kwargs
            if key in keys(te.params)
                set_param(te.params, key, val)
            else
                throw(ErrorException("Bad kwarg with key $(key) and value $(val) in TreeEnsemble constructor."))
            end
        end
        return te
    end
end

set_param(te::TreeEnsemble, s::Symbol, v::Any) = set_param(te.params, s, v)
get_param(te::TreeEnsemble, s::Symbol) = get_param(te.params, s)

""" Plants a set number of MIOTrees in a TreeEnsemble. """
function plant_trees(te::TreeEnsemble, n_trees::Int)
    solver = te.solver
    for i = 1:n_trees
        push!(te.trees, MIOTree(solver; te.params...))
    end
end

function train_ensemble(te::TreeEnsemble, X::Matrix, Y::Array)
    n_points = Int(floor(length(Y) / length(te.trees)))
    @showprogress 1 "Training ensemble of $(length(te.trees)) trees. " for i = 1:length(te.trees)
        tree = te.trees[i]
        idxs = (i-1) * n_points + 1: i * n_points
        if i == length(te.trees)
            idxs = (i-1) * n_points + 1: length(Y)
        end
        generate_MIO_model(tree, X[idxs,:], Y[idxs])
        optimize!(tree)
        populate_nodes!(tree)
        prune!(tree)
    end
    return
end