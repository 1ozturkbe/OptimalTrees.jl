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
    mt = MIOTree(SOLVER_SILENT; max_depth = 4, minbucket = 0.03)
    df = load_irisdata()
    X = Matrix(df[:,1:4])
    Y =  Array(df[:, "class"])
    md = 3
    set_param(mt, :max_depth, md)
    set_param(mt, :minbucket, 0.05)
    generate_binary_tree(mt)
    generate_MIO_model(mt, X, Y)
    @test length(allleaves(mt)) == 2^md

    set_optimizer(mt, SOLVER_SILENT)
    optimize!(mt)

    # Practicing pruning the tree
    m = mt.model
    as = getvalue.(m[:a])
    d = getvalue.(m[:d])
    Lt = getvalue.(m[:Lt])
    ckt = getvalue.(m[:ckt])
    populate_nodes!(mt)
    prune!(mt)
    @test length(allleaves(mt)) == sum(ckt .!= 0)
    @test score(mt, X, Y) == sum(Lt) && complexity(mt) == sum(as .!= 0)
end

function test_hyperplanecart()
    mt = MIOTree(SOLVER_SILENT)
    df = binarize(load_irisdata())
    X = Matrix(df[:,1:4])
    Y =  Array(df[:, "class"])
    hyperplane_cart(mt, X, Y)
    @test score(mt, X, Y) == 1.
    nds = apply(mt, X)
    @test all(getproperty.(nds, :label) .== Y)
    @test all(isnothing.(getproperty.(allleaves(mt), :label)) .== false)
    leaves = allleaves(mt)
    not_leaves = [nd for nd in allnodes(mt) if !is_leaf(nd)]
    @test all(is_leaf.(not_leaves) .== false)
    @test all(isnothing.(getproperty.(not_leaves, :a)) .== false)
    @test all(isnothing.(getproperty.(leaves, :label)) .== false)
end

# test_binarynode()

# test_MIOTree()

# test_hyperplanecart()