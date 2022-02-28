"""
    MIOTree_defaults(kwargs...)

Contains default MIOTree parameters, and modifies them with kwargs.
"""
function MIOTree_defaults(kwargs...)
    d = Dict(:max_depth => 5,
        :cp => 1e-6,
        :minbucket => 0.01) # in seconds
    if !isempty(kwargs)
        for (key, value) in kwargs
            set_param(d, key, value)
        end
    end
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
    function MIOTree(solver; kwargs...)
        root = BinaryNode(1)
        mt = new(JuMP.Model(solver),
                 root,
                 MIOTree_defaults(),
                 nothing)
        for (key, val) in kwargs
            if key in keys(mt.params)
                set_param(mt, key, val)
            else
                throw(ErrorException("Bad kwarg with key $(key) and value $(val) in MIOTree constructor."))
            end
        end
        return mt
    end
end

get_param(mt::MIOTree, sym::Symbol) = get_param(mt.params, sym)

set_param(mt::MIOTree, sym::Symbol, val::Any) = set_param(mt.params, sym, val)

""" Returns all BinaryNodes of MIOTree. """
allnodes(mt::MIOTree) = [mt.root, alloffspring(mt.root)...]

""" Returns all leaf BinaryNodes of MIOTree. """
allleaves(mt::MIOTree) = [nd for nd in allnodes(mt) if is_leaf(nd)]

"""Returns the leaf nodes in which the data X fall. """
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

""" Makes predictions using a tree, based on data X. """
function predict(mt::MIOTree, X::Matrix)
    vals = [] # TODO: initialize empty array instead, based on types of labels in MIOTree. 
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
    return vals
end

""" Makes predictions based on X, using the MIOTree. """
apply(mt::MIOTree, X::DataFrame) = apply(mt, Matrix(X))
# TODO: improve this by making sure that the DataFrame labels are in the right order. 

""" 
    $(TYPEDSIGNATURES)

Returns the prediction accuracy of MIOTree. 
"""
function score(mt::MIOTree)
    if JuMP.termination_status(mt.model) == MOI.OPTIMIZE_NOT_CALLED
        throw(ErrorException("`score` must be called with X, Y data if the MIOTree has not been optimized."))
    else
        return sum(JuMP.getvalue.(mt.model[:Lt]))/size(mt.model[:z],1)
    end
end

function score(mt::MIOTree, X, Y)
    preds = predict(mt, X)
    return sum(preds .== Y)/length(Y)
end

"""
    $(TYPEDSIGNATURES)

Returns the number of nonzero hyperplane coefficients of the MIOTree. 
"""
function complexity(mt::MIOTree)
    scor = 0
    for node in allnodes(mt)
        if !is_leaf(node)
            nonzeros = count(node.a .!= 0)
            scor += nonzeros
        end
    end
    return scor
end

"""
    $(TYPEDSIGNATURES)

Populates the nodes of the MIOTree using optimal solution of the MIO problem. 
"""
function populate_nodes!(mt::MIOTree)
    termination_status(mt.model) == MOI.OPTIMAL || 
        throw(ErrorException("MIOTree must be trained before it can be pruned."))
    queue = BinaryNode[mt.root]
    m = mt.model
    while !isempty(queue) # First populate the a,b hyperplane values
        nd = pop!(queue)
        if !is_leaf(nd)
            aval = getvalue.(m[:a][nd.idx, :])
            if sum(isapprox.(aval, zeros(length(aval)); atol = 1e-8)) != length(aval)
                nd.a = aval
                nd.b = getvalue.(m[:b][nd.idx])
                for child in children(nd)
                    push!(queue, child)
                end
            end
        end
    end
    for lf in allleaves(mt) # Then populate the class values...
        class_values = [isapprox(getvalue.(m[:ckt][i, lf.idx]), 1; atol=1e-8) for i = 1:length(mt.classes)]
        if sum(class_values) == 1
            lf.label = mt.classes[findall(class_values)[1]]
        elseif sum(class_values) > 1
            throw(ErrorException("Multiple classes assigned to node $(lf.idx)."))
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
    queue = BinaryNode[mt.root]
    while !isempty(queue)
        nd = pop!(queue)
        if !isnothing(nd.a) && any(nd.a != 0)
            for child in children(nd)
                push!(queue, child)
            end
        else
            alloffspr = alloffspring(nd)
            if isnothing(nd.label)
                alllabels = [nextnode.label for nextnode in alloffspr if !isnothing(nextnode.label)]
                if length(alllabels) == 1 
                    nd.label = alllabels[1]

                elseif length(alllabels) > 1
                    throw(ErrorException("Too many labels below node $(nd.idx)! Bug!"))
                else
                    throw(ErrorException("Missing labels below node $(nd.idx)! Bug!"))
                end
            end
            delete_children!(nd)
        end
    end
    # Add checks here so that the number of branches + leaves == total number of nodes
    return
end

""" Prunes the MIOTree from the root down, so that its offspring are garbage-collected. """
chop_down!(mt::MIOTree) = delete_children!(mt.root) # TODO: check garbage collection. 

function generate_binary_tree(mt::MIOTree)
    isempty(alloffspring(mt.root)) || throw(ErrorException("The MIOTree must be ungrown/chopped down to generate a dense binary tree. "))
    generate_binary_tree(mt.root, get_param(mt, :max_depth))
    return
end