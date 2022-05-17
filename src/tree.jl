"""
    MIOTree_defaults(kwargs...)

Contains default MIOTree parameters, and modifies them with kwargs.
"""
function MIOTree_defaults(;kwargs...)
    d = Dict(:max_depth => 5,
        :cp => 1e-6,
        :hypertol => 0.005, # hyperplane separation tolerance
        :minbucket => 0.01, 
        :regression_sparsity => :all, 
        :hyperplane_sparsity => :all, 
        :regression => false)
    set_params!(d; kwargs...)
    return d
end

""" 
    $(TYPEDSIGNATURES)

Returns nodes of a dense binary tree starting from root.
"""
function generate_binary_tree(root::BinaryNode, max_depth::Int)
    levs = Dict{Int64, Array{BinaryNode}}(0 => [root])
    nodes = [root]
    for i = 1:max_depth
        idxs = levs[i-1][end].idx .+ collect(1:2^i)
        levs[i] = BinaryNode[]
        for j = 1:length(levs[i-1])
            lc = BinaryNode(idxs[2*j-1])
            leftchild(levs[i-1][j], lc)
            push!(levs[i], lc)
            push!(nodes, lc)
            rc = BinaryNode(idxs[2*j])
            rightchild(levs[i-1][j], rc)
            push!(levs[i], rc)
            push!(nodes, rc)
        end
    end
    return nodes
end
    
mutable struct MIOTree
    model::JuMP.Model
    root::BinaryNode
    params::Dict
    classes::Union{Nothing, Array{Any}}
    solver

    function MIOTree(solver; kwargs...)
        root = BinaryNode(1)
        mt = new(JuMP.Model(solver),
                 root,
                 MIOTree_defaults(),
                 nothing,
                 solver)
        for (key, val) in kwargs
            if key in keys(mt.params)
                set_param!(mt, key, val)
            else
                throw(ErrorException("Bad kwarg with key $(key) and value $(val) in MIOTree constructor."))
            end
        end
        return mt
    end
end

get_param(mt::MIOTree, sym::Symbol) = get_param(mt.params, sym)

set_param!(mt::MIOTree, sym::Symbol, val::Any) = set_param!(mt.params, sym, val)

set_params!(mt::MIOTree; kwargs...) = set_params!(mt.params; kwargs...)

"""
    $(TYPEDSIGNATURES)

Clones an MIOTree via deepcopy. 
"""
function clone(mt::MIOTree)
    return deepcopy(mt)
end

""" Returns all BinaryNodes of MIOTree. """
allnodes(mt::MIOTree) = [mt.root, alloffspring(mt.root)...]

""" Returns all leaf BinaryNodes of MIOTree. """
allleaves(mt::MIOTree) = [nd for nd in allnodes(mt) if is_leaf(nd)]

"""
    $(TYPEDSIGNATURES)

Returns the leaf nodes in which the data X fall. 
"""
function apply(mt::MIOTree, X::Matrix)
    vals = BinaryNode[] # TODO: initialize empty array instead, based on types of indices in MIOTree. 
    for i = 1:size(X, 1)
        row = X[i,:]
        nd = mt.root
        while !is_leaf(nd)
            lhs = sum(nd.a .* row)
            if lhs ≤ nd.b
                nd = nd.left
            else
                nd = nd.right
            end
        end
        push!(vals, nd)
    end
    return vals
end

""" 
    $(TYPEDSIGNATURES)

Makes predictions using a tree, based on data X. 
"""
function predict(mt::MIOTree, X::Matrix)
    vals = [] # TODO: initialize empty array instead, based on types of labels in MIOTree. 
    if get_param(mt, :regression)
        for i = 1:size(X, 1)
            row = X[i,:]
            nd = mt.root
            while !is_leaf(nd)
                lhs = sum(nd.a .* row)
                if lhs ≤ nd.b
                    nd = nd.left
                else
                    nd = nd.right
                end
            end
            push!(vals, nd.label[1] + sum(nd.label[2] .* row))
        end
    else
        for i = 1:size(X, 1)
            row = X[i,:]
            nd = mt.root
            while !is_leaf(nd)
                lhs = sum(nd.a .* row)
                if lhs ≤ nd.b
                    nd = nd.left
                else
                    nd = nd.right
                end
            end
            push!(vals, nd.label)
        end
    end
    return vals
end

apply(mt::MIOTree, X::DataFrame) = apply(mt, Matrix(X))
# TODO: improve this by making sure that the DataFrame labels are in the right order. 

""" 
    $(TYPEDSIGNATURES)

Returns the prediction accuracy of MIOTree. 
For classification: misclassification accuracy.
For regression: R^2.  
"""

function score(mt::MIOTree, X, Y)
    preds = predict(mt, X)
    if get_param(mt, :regression)
        return 1 - sum((preds .- Y).^2) / sum((preds .- sum(Y)/length(Y)*ones(length(Y))).^2)
    else
        return sum(preds .== Y)/length(Y)
    end
end

"""
    $(TYPEDSIGNATURES)

Returns the number of nonzero hyperplane coefficients of the MIOTree. 
"""
function complexity(mt::MIOTree)
    scor = 0
    for node in allnodes(mt)
        if !is_leaf(node) && !isnothing(node.a)
            nonzeros = length(node.a) - sum(isapprox.(node.a, zeros(length(node.a))))
            scor += nonzeros
        end
    end
    return scor
end

""" 
    $(TYPEDSIGNATURES)

Cleans all variables and constraints from a MIOTree. 
"""
function clean_model!(mt::MIOTree)
    mt.model = JuMP.Model(mt.solver)
    return
end

"""
    $(TYPEDSIGNATURES)

Populates the nodes of the MIOTree using optimal solution of the MIO problem. 
"""
function populate_nodes!(mt::MIOTree)
    termination_status(mt.model) == MOI.OPTIMAL || 
        throw(ErrorException("MIOTree must be trained before it can be pruned."))
    m = mt.model
    for nd in allnodes(mt)
        if !is_leaf(nd)
            aval = value.(m[:a][nd.idx, :])
            if sum(isapprox.(aval, zeros(length(aval)); atol = 1e-10)) != length(aval)
                nd.a = aval
                nd.b = value.(m[:b][nd.idx])
            end
        end
    end
    if get_param(mt, :regression)
        for lf in allleaves(mt)
            regr_coeffs = value.(m[:beta][lf.idx, :])
            regr_const = value(m[:beta0][lf.idx])
            set_classification_label!(lf, (regr_const, regr_coeffs))
        end
    else
        for lf in allleaves(mt) # Then populate the class values...
            class_values = [isapprox(value.(m[:ckt][i, lf.idx]), 1; atol=1e-4) for i = 1:length(mt.classes)]
            if sum(class_values) == 1
                set_classification_label!(lf, mt.classes[findall(class_values)[1]])
            elseif sum(class_values) > 1
                throw(ErrorException("Multiple classes assigned to node $(lf.idx)."))
            end
        end
    end
    return
end

""" 
    prune!(mt::MIOTree)

Prunes MIOTree depending on hyperplane coefficients and class values. 
See ```populate_nodes''' for 
how to populate nodes using optimal solution data. 
"""
function prune!(mt::MIOTree)
    queue = allleaves(mt)
    while !isempty(queue) # A bottom-up approach for pruning. 
        node = popfirst!(queue)
        par = node.parent
        if is_leaf(node) && isnothing(node.label) && !isnothing(node.parent)
            offspring = alloffspring(par) # TODO: speed this up
            labels = [of.label for of in offspring if !isnothing(of.label)]
            if length(labels) == 1
                delete_children!(par)
                set_classification_label!(par, labels[1])
            elseif length(labels) == 0
                delete_children!(par)
                push!(queue, par)
            end
        end
    end
    leaves = allleaves(mt)
    if any(isnothing(leaf.label) for leaf in leaves)
        @warn "A subset of leaves is unlabeled. Must label for correct predictions. "
    end
    return
end

""" Prunes the MIOTree from the root down, so that its offspring are garbage-collected. """
chop_down!(mt::MIOTree) = delete_children!(mt.root) # TODO: check garbage collection. 

function generate_binary_tree(mt::MIOTree)
    isempty(alloffspring(mt.root)) || throw(ErrorException("The MIOTree must be ungrown/chopped down to generate a dense binary tree. "))
    generate_binary_tree(mt.root, get_param(mt, :max_depth))
    return
end