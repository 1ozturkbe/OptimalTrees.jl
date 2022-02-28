#  Let's build tree structure (borrowed from AbstractTrees.jl)
@with_kw mutable struct BinaryNode{T}
    idx::T
    parent::Union{BinaryNode{T}, Nothing} = nothing
    left::Union{BinaryNode{T}, Nothing} = nothing 
    right::Union{BinaryNode{T}, Nothing} = nothing
    a::Union{Array, Nothing} = nothing 
    b::Union{Real, Nothing} = nothing
    label::Any = nothing
end
BinaryNode(idx) = BinaryNode{typeof(idx)}(idx = idx)
BinaryNode(idx, parent::BinaryNode) = BinaryNode{typeof(idx)}(idx = idx, parent = parent)

function Base.show(io::IO, bn::BinaryNode)
    if is_leaf(bn)
        print("Leaf BinaryNode $(bn.idx) with label $(string(bn.label)).")
    else
        print("Branching BinaryNode $(bn.idx).")
    end
    return
end

""" 
    $(TYPEDSIGNATURES)

Adds a left (less-than) child to BinaryNode. 
"""
function leftchild(parent::BinaryNode, child::BinaryNode)
    isnothing(parent.left) || error("Left child of node $(parent.idx) is already assigned.")
    isnothing(child.parent) || error("Parent of node $(child.idx) is already assigned.")
    parent.left = child    
    child.parent = parent
    return
end

""" 
    $(TYPEDSIGNATURES)

Adds a right (greater-than) child to BinaryNode. 
"""
function rightchild(parent::BinaryNode, child::BinaryNode)
    isnothing(parent.right) || error("Right child of node $(parent.idx) is already assigned.")
    isnothing(child.parent) || error("Parent of node $(child.idx) is already assigned.")
    parent.right = child    
    child.parent = parent
    return
end

""" 
    children(node::BinaryNode)

Returns (left, right) children of a BinaryNode.
"""
function children(node::BinaryNode)
    if !isnothing(node.left)
        if !isnothing(node.right)
            return (node.left, node.right)
        end
        return (node.left,)
    end
    !isnothing(node.right) && return (node.right,)
    return ()
end

""" Returns all "younger" relatives of a BinaryNode. """
function alloffspring(nd::BinaryNode)
    offspr = [child for child in children(nd)]
    queue = [child for child in children(nd)]
    while !isempty(queue)
        nextnode = popfirst!(queue)
        if !is_leaf(nextnode)
            for child in children(nextnode)
                push!(offspr, child)
                push!(queue, child)
            end
        else
            for child in children(nextnode)
                push!(offspr, child)
            end
        end
    end
    return offspr
end

"""
    delete_children!(bn::BinaryNode)

"Deletes" the children of BinaryNode by pruning and returning the nodes.   
"""
function delete_children!(bn::BinaryNode)
    allchildren = children(bn)
    bn.left = nothing
    bn.right = nothing
    for child in allchildren
        child.parent = nothing
    end
end

""" Better node printing."""
printnode(io::IO, node::BinaryNode) = print(io, node.idx)

""" 
Returns a-coefficients and b-value of the split on BinaryNode. 
"""
function get_split_values(bn::BinaryNode)
    is_leaf(bn) && throw(ErrorException("Cannot get split values of leaf node $(bn.idx)."))
    return bn.a, bn.b
end

""" 
Sets a-coefficients and b-value of a split on BinaryNode. 
"""
function set_split_values!(bn::BinaryNode, a, b)
    is_leaf(bn) && throw(ErrorException("Cannot set split values for leaf node $(bn.idx)."))
    bn.a = a
    bn.b = b
    bn.label = nothing
    return
end

""" 
    is_leaf(bn::BinaryNode)

Checks whether a BinaryNode is a leaf of the tree. 
"""
function is_leaf(bn::BinaryNode)
    return isnothing(bn.left) && isnothing(bn.right)
end

"""
    get_classification_label(bn::BinaryNode)

Returns the classification label of a leaf BinaryNode. 
"""
function get_classification_label(bn::BinaryNode)
    is_leaf(bn) || throw(ErrorException("Cannot get the classification label of node $(bn.idx), " * 
                        "since it is not a leaf node."))
    return bn.label
end

"""
    set_classification_label!(bn::BinaryNode)

Sets the classification label of a leaf BinaryNode. 
"""
function set_classification_label!(bn::BinaryNode, label)
    is_leaf(bn) || throw(ErrorException("Cannot set the classification label of node $(bn.idx), " * 
    "since it is not a leaf node."))
    bn.label = label
    bn.a = nothing
    bn.b = nothing
    return
end
