""" 
    $(TYPEDSIGNATURES)
    
Finds all leaves of MIOTree.
"""
find_leaves(mt::MIOTree) = [nd for nd in allnodes(mt) if is_leaf(nd)]

""" Checks if a MIOTree is trained. """
function check_if_trained(mt::MIOTree)
    for nd in allnodes(mt)
        (is_leaf(nd) && isnothing(nd.label)) && return false
        (!is_leaf(nd) && (isnothing(nd.a) || isnothing(nd.b))) && return false
    end
    return true
end

""" Prints problems with MIOTree for debugging. """
function debug_if_trained(mt::MIOTree)
    for nd in allnodes(mt)
        (is_leaf(nd) && isnothing(nd.label)) && println("Leaf $(nd.idx) has no label.")
        (!is_leaf(nd) && (isnothing(nd.a) || isnothing(nd.b))) && println("Split $(nd.idx) has no hyperplane data.")
    end
end

"""
    $(TYPEDSIGNATURES)

Returns the hyperplane data from an MIOTree,
in the format Dict[leaf_number] containing [B0, B]. 
"""
function trust_region_data(mt::MIOTree)
    all_leaves = find_leaves(mt)
    upperDict = Dict()
    lowerDict = Dict()
    for leaf in all_leaves
        line = [leaf, lineage(leaf)...]
        ups = [line[i] for i = 2:length(line) if line[i-1] == line[i].left]
        lows = [line[i] for i = 2:length(line) if line[i-1] == line[i].right]
        upperDict[leaf.idx] = [[nd.b, nd.a] for nd in ups]
        lowerDict[leaf.idx] = [[nd.b, nd.a] for nd in lows]
    end
    return upperDict, lowerDict
end

"""
    $(TYPEDSIGNATURES)

Returns the regression weights from an OptimalTreeLearner by leaf, 
i.e. Dict[leaf_number] containing [B0, B].
"""
function pwl_constraint_data(lnr::MIOTree, vks)
    @assert get_param(lnr, :regression)
    return Dict(leaf.idx => leaf.label for leaf in find_leaves(lnr))
end


function deepen_to_max_depth!(mt::MIOTree)
    md = get_param(mt, :max_depth)
    queue = allnodes(mt)
    idx = maximum([nd.idx for nd in queue])
    while !isempty(queue)
        node = popfirst!(queue)
        if is_leaf(node) && depth(node) < md
            idx += 1
            leftchild(node, BinaryNode(idx))
            idx += 1
            rightchild(node, BinaryNode(idx))
            append!(queue, [node.left, node.right])
        end
    end
    return
end

function deepen_one_level!(mt::MIOTree)
    md = get_param(mt, :max_depth)
    queue = allnodes(mt)
    idx = maximum([nd.idx for nd in queue])
    while !isempty(queue)
        node = popfirst!(queue)
        if is_leaf(node) && depth(node) < md
            idx += 1
            leftchild(node, BinaryNode(idx))
            idx += 1
            rightchild(node, BinaryNode(idx))
        end
    end
    return
end