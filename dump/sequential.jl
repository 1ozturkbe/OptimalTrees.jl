
""" Adds one layer of depth to MIOTree."""
function add_depth(mt::MIOTree)
    all_nodes = [mt.root, alloffspring(mt.root)...]
    all_leaves = [nd for nd in all_nodes if is_leaf(nd)]
    max_idx = maximum([nd.idx for nd in all_nodes])
    all_idxs = max_idx .+ collect(1:2*length(all_leaves))
    new_leaves = BinaryNode[]
    for leaf in all_leaves
        leftchild(leaf, BinaryNode(popfirst!(all_idxs)))
        push!(new_leaves, leaf.left)        
        rightchild(leaf, BinaryNode(popfirst!(all_idxs)))
        push!(new_leaves, leaf.right)
    end
    return
end