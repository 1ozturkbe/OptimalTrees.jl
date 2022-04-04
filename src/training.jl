JuMP.set_optimizer(mt::MIOTree, solver) = JuMP.set_optimizer(mt.model, solver)

JuMP.optimize!(mt::MIOTree) = JuMP.optimize!(mt.model)

"""
    generate_MIO_model(mt::MIOTree, X::Matrix, Y::Array)

Generates a MIO model of the tree defined from mt.root, with data X and Y.
"""
function generate_MIO_model(mt::MIOTree, X::Matrix, Y::Array)
    n_samples, n_vars = size(X)

    # Reference minimal parameters
    nds = allnodes(mt)
    lfs = [nd for nd in nds if is_leaf(nd)]
    nd_idxs = getproperty.(nds, :idx) # Node indices
    lf_idxs = getproperty.(lfs, :idx) # Leaf indices
    sp_idxs = [idx for idx in nd_idxs if idx ∉ lf_idxs]
    min_points = get_param(mt, :minbucket)
    if !isa(min_points, Int) && 0 <= min_points <= 1
        min_points = Int(ceil(min_points * n_samples))
    else
        throw(ErrorException("Minbucket parameter must be between 0-1 or an integer!"))
    end
    mt.classes = sort(unique(Y)) # The potential classes are sorted. 
    k = length(mt.classes)

    @variable(mt.model, -1 <= a[sp_idxs, 1:n_vars] <= 1)
    @variable(mt.model, 0 <= abar[sp_idxs, 1:n_vars])
    @variable(mt.model, -1 <= b[sp_idxs] <= 1)
    @variable(mt.model, d[sp_idxs], Bin)
    @variable(mt.model, s[sp_idxs, 1:n_vars], Bin) # Binary variables for complexity penalty
    @constraint(mt.model, -b .<= d)
    @constraint(mt.model, b .<= d)
    @constraint(mt.model, -a .<= s)
    @constraint(mt.model, a .<= s)
    @constraint(mt.model, a .<= abar)
    @constraint(mt.model, -a .<= abar)
    @constraint(mt.model, [j=1:n_vars], s[:,j] .<= d[:])
    @constraint(mt.model, [i = sp_idxs], sum(s[i, :]) >= d[i])
    @constraint(mt.model, [i = sp_idxs, j=1:n_vars], a[i,j] <= s[i,j])
    @constraint(mt.model, [i = sp_idxs, j=1:n_vars], -s[i,j] <= a[i,j])
    
    # Enforcing each point to one leaf
    @variable(mt.model, z[1:n_samples, lf_idxs], Bin)
    @constraint(mt.model, [i=1:n_samples], sum(z[i, :]) == 1)

    # Making sure that variables are properly binned. 
    @variable(mt.model, ckt[1:k, lf_idxs], Bin)  # Class at leaf
    @variable(mt.model, Nt[lf_idxs] >= 0)       # Total number of points at leaf
    @variable(mt.model, lt[lf_idxs], Bin)       # Whether or not a leaf is occupied
    @variable(mt.model, Nkt[1:k, lf_idxs] >= 0) # Number of points of at leaf with class k

    @constraint(mt.model, [i = lf_idxs], Nt[i] == sum(z[:, i])) # Counting number of points in a leaf. 
    @constraint(mt.model, [i = lf_idxs], sum(ckt[:, i]) == lt[i]) # Making sure a class is only assigned if leaf is occupied.
    for kn = 1:k
        # Number of values of each class
        @constraint(mt.model, [i = lf_idxs], Nkt[kn, i] == 
                    sum(z[l, i] for l = 1:n_samples if Y[l] == mt.classes[kn]))
    end

    # Loss function
    @variable(mt.model, Lt[lf_idxs] >= 0)
    @constraint(mt.model, [i = lf_idxs, j = 1:k], Lt[i] >= Nt[i] - Nkt[j, i] - n_samples * (1-ckt[j,i]))
    @constraint(mt.model, [i = lf_idxs, j = 1:k], Lt[i] <= Nt[i] - Nkt[j, i] + n_samples * ckt[j,i])

    @constraint(mt.model, sum(abar[mt.root.idx, :]) <= d[mt.root.idx])
    for nd in nds
        if !is_leaf(nd) && !isnothing(nd.parent)
            @constraint(mt.model, d[nd.idx] <= d[nd.parent.idx])
            @constraint(mt.model, sum(abar[nd.idx, :]) <= d[nd.idx])
        end
    end

    mu = get_param(mt, :hypertol) # hyperplane separation tolerance
    for lf in lfs
        # Enforcing minbucket 
        @constraint(mt.model, [i=1:n_samples], z[i, lf.idx] <= lt[lf.idx])
        @constraint(mt.model, sum(z[:, lf.idx]) >= min_points*lt[lf.idx])
        # Enforcing hyperplane splits
        for i=1:n_samples
            ps = [lf, lineage(lf)...] # the parent sequence
            for j = 1:length(ps)-1
                if ps[j].idx == ps[j+1].left.idx
                    @constraint(mt.model, sum(a[ps[j+1].idx, :] .* X[i, :]) + mu <= b[ps[j+1].idx] + (2+mu)*(1-z[i,lf.idx])) 
                elseif ps[j].idx == ps[j+1].right.idx
                    @constraint(mt.model, sum(a[ps[j+1].idx, :] .* X[i, :]) >= b[ps[j+1].idx] - 2*(1-z[i,lf.idx])) 
                else
                    throw(ErrorException("Node backtracking failed for some reason. Bug."))
                end
            end
        end
    end

    # Objective function: misclassification error + complexity (depth * label cost).
    # Increasing penalty by depth to ensure that splits are created top-down and there are no discontinuities in the tree.
    @objective(mt.model, Min, 1/n_samples * sum(Lt) + get_param(mt, :cp) * 
            (sum(depth(nd)*(sum(s[nd.idx,:]) + d[nd.idx]) for nd in nds if !is_leaf(nd)))        )
    return
end

""" 
    SVM(X::Matrix, Y::Array, solver, C = 0.01)

Optimizes an SVM, where C is the regularization factor. 
"""
function SVM(X::Matrix, Y::Array, solver, C = 0.01)
    n_samples, n_vars = size(X)
    classes = sort(unique(Y)) # The potential classes are sorted. 
    k = length(classes)
    k == 2 || throw(ErrorException("Detected $(k) class for the SVM." * 
        " Only two classes allowed."))
    Y_san = ones(n_samples) # TODO: maybe don't generate -1/1 data every time? 
    Y_san[findall(Y .== classes[1])] .= -1
    m = JuMP.Model(solver)
    @variable(m, a[1:n_vars])
    @variable(m, b)
    @variable(m, ζ[1:n_samples] ≥ 0) # variable allocation 
    @constraint(m, [i = 1:n_samples], Y_san[i] * (sum(a.*X[i, :]) - b) ≥ 1 - ζ[i])
    C = 100
    @objective(m, Min, 0.5*C*sum(a.^2) + sum(ζ))
    optimize!(m)
    return getvalue.(a), getvalue(b)
end 

"""
    $(TYPEDSIGNATURES)

Performs greedy tree training with hyperplanes for binary classification.
"""
function hyperplane_cart(mt::MIOTree, X::Matrix, Y::Array)
    n_samples, n_vars = size(X)
    classes = sort(unique(Y))
    length(classes) == 2 || throw(ErrorException("Hyperplane CART can only be applied to binary classification problems. "))
    isempty(alloffspring(mt.root)) || throw(ErrorException("Hyperplane CART can only be applied to ungrown trees. "))
    minpoints = ceil(n_samples * get_param(mt, :minbucket))
    counts = Dict((i => count(==(i), Y)) for i in unique(Y))
    maxval = 0
    maxkey = ""
    for (key, val) in counts
        if val ≥ maxval
            maxval = val
            maxkey = key
        end
    end
    valid_leaves = [mt.root] # Stores leaves ready for SVM cuts. 
    ct = 1
    point_idxs = Dict(mt.root.idx => collect(1:n_samples))
    while !isempty(valid_leaves)
        leaf = popfirst!(valid_leaves)
        leaf.a, leaf.b = SVM(X[point_idxs[leaf.idx], :],
                            Y[point_idxs[leaf.idx]], mt.solver)

        # Checking left child, and adding to valid_leaves if necessary
        ct += 1
        leftchild(leaf, BinaryNode(ct))
        left_idxs = findall(x -> x <= 0, 
            [sum(leaf.a .*X[i, :]) - leaf.b for i = point_idxs[leaf.idx]])
        point_idxs[leaf.left.idx] = point_idxs[leaf.idx][left_idxs]
        n_pos_left = count(Y[left_idxs].== classes[2])
        n_neg_left = count(Y[left_idxs] .== classes[1])

        # Checking right child, and adding to valid leaves if necessary
        ct += 1
        rightchild(leaf, BinaryNode(ct))
        right_idxs = findall(x -> x > 0, 
        [sum(leaf.a .*X[i, :]) - leaf.b for i = point_idxs[leaf.idx]])
        point_idxs[leaf.right.idx] = point_idxs[leaf.idx][right_idxs]
        n_pos_right = count(Y[right_idxs] .== classes[2])
        n_neg_right = count(Y[right_idxs] .== classes[1])

        # Setting labels
        if n_pos_left/n_neg_left > 1 
            set_classification_label!(leaf.left, classes[2])
        elseif n_pos_left/n_neg_left == 1
            @warn "Leaf $(leaf.left.idx) has samples that are split 50/50. Will set based on label of sibling."
        else
            set_classification_label!(leaf.left, classes[1])
        end
        if n_pos_right/n_neg_right > 1 
            set_classification_label!(leaf.right, classes[2])
        elseif n_pos_right/n_neg_right == 1
            @warn "Leaf $(leaf.right.idx) has samples that are split 50/50. Will set based on label of sibling."
        else
            set_classification_label!(leaf.right, classes[1])
        end
        # Resolving leaf labels based on sibling leaves. 
        if isnothing(leaf.left.label) 
            if isnothing(leaf.right.label)
                throw(ErrorException("Data seems to be perfectly random. Bug."))
            else
                leaf.left.label = findall(x -> x != leaf.right.label, classes)[1]
            end
            leaf.right.label = findall(x -> x != leaf.left.label, classes)[1]
        end
        
        # Pruning if necessary, 
        # and choosing whether leaves should be added to queue. 
        if leaf.left.label == leaf.right.label
            point_idxs[leaf.idx] = union(point_idxs[leaf.left.idx], point_idxs[leaf.right.idx])
            delete!(point_idxs, leaf.left.idx)
            delete!(point_idxs, leaf.right.idx)
            parent_label = leaf.left.label
            delete_children!(leaf)
            leaf.label = parent_label
        else
            left_accuracy = abs(n_pos_left - n_neg_left) / length(left_idxs)              
            right_accuracy = abs(n_pos_right - n_neg_right) / length(right_idxs)
            if left_accuracy != 1 && 1 - left_accuracy ≥ minpoints / length(left_idxs)
                push!(valid_leaves, leaf.left)
            end
            if right_accuracy != 1 && 1 - right_accuracy ≥ minpoints / length(right_idxs)
                push!(valid_leaves, leaf.right)
            end
            # Since leaf is no longer a leaf, deleting its point indices. 
            delete!(point_idxs, leaf.idx)
        end
    end
    return
end

""" Warmstarts an MIOTree model using the last solution stored in its nodes. """
function warmstart(mt::MIOTree)
    m = mt.model
    for nd in allnodes(mt)
        if is_leaf(nd) && !isnothing(nd.label)
            for i = 1:length(mt.classes)
                if mt.classes[i] == nd.label
                    JuMP.set_start_value(m[:ckt][i, nd.idx], 1)
                else
                    JuMP.set_start_value(m[:ckt][i, nd.idx], 0)
                end
            end
        elseif !isnothing(nd.a)
            JuMP.set_start_value.(m[:a][nd.idx, :], nd.a)
            JuMP.set_start_value(m[:b][nd.idx], nd.b)
        end
    end
    return
end

""" Trains a tree sequentially by increasing its depth. """
function sequential_train!(mt::MIOTree, X::Matrix, Y::Array, min_depth::Integer = 1; pruning = false)
    max_depth = get_param(mt, :max_depth)
    md = min_depth
    set_param(mt, :max_depth, min_depth)
    chop_down!(mt)
    generate_binary_tree(mt)
    clean_model!(mt)
    generate_MIO_model(mt, X, Y)
    optimize!(mt)
    populate_nodes!(mt)
    pruning && prune!(mt)
    while md < max_depth
        md += 1
        set_param(mt, :max_depth, md)
        deepen_one_level!(mt)
        clean_model!(mt)
        generate_MIO_model(mt, X, Y)
        warmstart(mt)
        optimize!(mt)
        populate_nodes!(mt)
        pruning && prune!(mt)
    end
    set_param(mt, :max_depth, max_depth)
    return
end