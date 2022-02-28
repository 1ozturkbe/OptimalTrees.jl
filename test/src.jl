function load_irisdata()
    path = "data/iris.data"
    csv_data = CSV.File(path, header=false)
    iris_names = ["sepal_len", "sepal_wid", "petal_len", "petal_wid", "class"]
    df = DataFrame(csv_data)
    rename!(df, iris_names)
    dropmissing!(df)
    return df
end

""" Turns classes in DataFrame (last column) into binary classes. """
function binarize(df = load_irisdata())
    df_bin = copy(df)
    X = df_bin[:,1:end-1]
    Y = df_bin[:, end]
    classes = unique(Y)
    for i = 1:length(Y)
        if Y[i] != classes[1]
            Y[i] = classes[2]
        end
    end
    df_bin[:, end] = Y
    return df_bin
end

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

""" Tests non-optimization functionalities of MIOTree. """
function test_MIOTree()
    d = MIOTree_defaults()
    d = MIOTree_defaults(:max_depth => 4, :cp => 1e-5)
    @test d[:max_depth] == 4
    @test get_param(d, :cp) == 1e-5
    mt = MIOTree(SOLVER_SILENT; max_depth = 2, minbucket = 0.03)
    set_param(mt, :max_depth, 4)
    df = load_irisdata()
    X = Matrix(df[:,1:4])
    Y =  Array(df[:, "class"])
    md = 3
    set_param(mt, :max_depth, md)
    set_param(mt, :minbucket, 0.001)
    generate_binary_tree(mt)
    generate_MIO_model(mt, X, Y)
    @test length(allleaves(mt)) == 2^md
end

""" Tests full MIO-based tree learning. """
function test_MIOtraining(df = load_irisdata())
    mt = MIOTree(SOLVER_SILENT)
    df = binarize(load_irisdata())
    X = Matrix(df[:,1:4])
    Y =  Array(df[:, "class"])
    md = 3
    set_param(mt, :max_depth, md)
    set_param(mt, :minbucket, 0.001)
    generate_binary_tree(mt)
    generate_MIO_model(mt, X, Y)
    set_optimizer(mt, SOLVER_SILENT)
    optimize!(mt)
    # # Practicing pruning the tree
    m = mt.model
    as = getvalue.(m[:a])
    d = getvalue.(m[:d])
    Lt = getvalue.(m[:Lt])
    ckt = getvalue.(m[:ckt])
    populate_nodes!(mt)
    prune!(mt)
    @test length(allleaves(mt)) == sum(ckt .!= 0)
    @test accuracy(mt) == sum(Lt) && complexity(mt) == sum(as .!= 0)
end

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
        @info "Trying $(leaf.idx)..."
        leaf.a, leaf.b = SVM(X[point_idxs[leaf.idx], :],
                            Y[point_idxs[leaf.idx]], SOLVER_SILENT)

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

function test_hyperplanecart()
    @test true
end

test_binarynode()

test_MIOTree()

test_MIOtraining()

test_hyperplanecart()