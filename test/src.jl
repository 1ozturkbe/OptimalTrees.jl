""" Tests the functionalities of BinaryNode. """
function test_binarynode()
    bn = BinaryNode(1)
    leftchild(bn, BinaryNode(2))
    rightchild(bn, BinaryNode(3))
    rightchild(bn.right, BinaryNode(4))
    @test all([child.idx for child in children(bn)] .== [2,3]) &&
        isempty(children(bn.right.right)) && length(alloffspring(bn)) == 3
    
    delete_children!(bn.right)
    @test isnothing(bn.parent) && isnothing(bn.right.right) && 
        isempty(children(bn.right)) && length(alloffspring(bn)) == 2

    @test_throws ErrorException set_split_values!(bn.right, [1,2,3], 4)
    @test_throws ErrorException set_classification_label!(bn, 5)
end



""" Tests MIO-based tree learning. """
function test_training()
    path = "data/iris.data"
    csv_data = CSV.File(path, header=false)
    iris_names = ["sepal_len", "sepal_wid", "petal_len", "petal_wid", "class"]
    df = DataFrame(csv_data)
    rename!(df, iris_names)
    dropmissing!(df)



    # Checking MIOTree
    d = MIOTree_defaults()
    d = MIOTree_defaults(:max_depth => 4, :cp => 1e-5)
    @test d[:max_depth] == 4
    @test get_param(d, :cp) == 1e-5
    mt = build_MIOTree(CPLEX_SILENT; max_depth = 2, minbucket = 0.03)
    set_param(mt, :max_depth, 4)

    X = Matrix(df[:,1:4])
    Y =  Array(df[:, "class"])
    md = 3
    set_param(mt, :max_depth, md)
    set_param(mt, :minbucket, 0.001)
    generate_tree_model(mt, X, Y)
    @test all(is_leaf.(mt.leaves))
    @test sum(is_leaf.(mt.nodes)) == 2^md
    set_optimizer(mt, CPLEX_SILENT)
    optimize!(mt)
    # # Practicing pruning the tree
    m = mt.model
    as = getvalue.(m[:a])
    bs = getvalue.(m[:b])
    Nkt = getvalue.(m[:Nkt])
    Nt = getvalue.(m[:Nt])
    d = getvalue.(m[:d])
    Lt = getvalue.(m[:Lt])
    ckt = getvalue.(m[:ckt])
    populate_nodes!(mt)
    prune!(mt)
    @test length(mt.nodes) == length(alloffspring(mt.root)) + 1 
    @test length(mt.leaves) == sum(ckt .!= 0)
    @test score(mt) == sum(Lt) && complexity(mt) == sum(as .!= 0)

    # We know there exists a perfect classifier, so let's test apply
    # @test all(apply(mt, X) .== Y)
    # For some reason, optimal trees don't seem to be optimal...
    # FIgure out why I'm getting non-zero 
end

# test_training()

""" Adds one layer of depth to MIOTree."""
function add_depth(mt::MIOTree)
    max_idx = maximum([nd.idx for nd in mt.nodes])
    all_idxs = max_idx .+ collect(1:2*length(mt.leaves))
    new_leaves = BinaryNode[]
    for leaf in mt.leaves
        leftchild(leaf, BinaryNode(popfirst!(all_idxs)))
        push!(new_leaves, leaf.left)        
        rightchild(leaf, BinaryNode(popfirst!(all_idxs)))
        push!(new_leaves, leaf.right)
    end
    mt.leaves = new_leaves
    return
end

mt = build_MIOTree(CPLEX_SILENT; max_depth = 6, minbucket = 0.03)
# Sanitize Y for SVM
path = "data/iris.data"
csv_data = CSV.File(path, header=false)
iris_names = ["sepal_len", "sepal_wid", "petal_len", "petal_wid", "class"]
df = DataFrame(csv_data)
rename!(df, iris_names)
dropmissing!(df)

X = Matrix(df[:,1:4])
Y = Array(df[:,5])
Y_bin = []
for val in Y
    if val == "Iris-setosa"
        push!(Y_bin, "Iris-versicolor")
    else
        push!(Y_bin, val)
    end
end
Y = Y_bin

function hyperplane_cart(mt::MIOTree, X::Matrix, Y::Array)
    n_samples, n_vars = size(X)
    classes = sort(unique(Y))
    minpoints = ceil(n_samples * get_param(mt, :minbucket))
    delete_children!(mt.root)
    mt.leaves = [mt.root]
    mt.nodes = [mt.root]
    counts = Dict((i => count(==(i), Y)) for i in unique(Y))
    maxval = 0
    maxkey = ""
    for (key, val) in counts
        global maxval
        global maxkey
        if val ≥ maxval
            maxval = val
            maxkey = key
        end
    end
    valid_leaves = [mt.root] # Stores leaves ready for SVM cuts. 
    ct = 1
    point_idxs = Dict(mt.root.idx => collect(1:n_samples))
    while !isempty(valid_leaves)
        global ct
        global point_idxs
        global valid_leaves
        leaf = popfirst!(valid_leaves)
        @info "Trying $(leaf.idx)..."
        leaf.a, leaf.b = SVM(X[point_idxs[leaf.idx], :],
                            Y[point_idxs[leaf.idx]], CPLEX_SILENT)

        # Checking left child, and adding to valid_leaves if necessary
        ct += 1
        leftchild(leaf, BinaryNode(ct))
        left_idxs = findall(x -> x <= 0, 
            [sum(leaf.a .*X[i, :]) - leaf.b for i = point_idxs[leaf.idx]])
        point_idxs[leaf.left.idx] = point_idxs[leaf.idx][left_idxs]
        n_pos_left = sum(Y[left_idxs].== classes[2])
        n_neg_left = sum(Y[left_idxs] .== classes[1])

        # Checking right child, and adding to valid leaves if necessary
        ct += 1
        rightchild(leaf, BinaryNode(ct))
        right_idxs = findall(x -> x > 0, 
        [sum(leaf.a .*X[i, :]) - leaf.b for i = point_idxs[leaf.idx]])
        point_idxs[leaf.right.idx] = point_idxs[leaf.idx][right_idxs]
        n_pos_right = sum(Y[right_idxs].== classes[2])
        n_neg_right = sum(Y[right_idxs] .== classes[1])

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
            left_error_rate = abs(n_pos_left - n_neg_left)/(2*(length(left_idxs)))                
            right_error_rate = abs(n_pos_right - n_neg_right)/(2*(length(right_idxs)))
            if left_error_rate != 0 && abs(n_pos_left - n_neg_left) ≥ minpoints
                push!(valid_leaves, leaf.left)
            end
            if right_error_rate != 0 && abs(n_pos_right - n_neg_right) ≥ minpoints
                push!(valid_leaves, leaf.right)
            end
            # Since leaf is no longer a leaf, deleting its point indices. 
            delete!(point_idxs, leaf.idx)
        end
    end
    return
end