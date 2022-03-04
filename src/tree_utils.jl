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


