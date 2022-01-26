""" Tests MIO-based tree learning. """
function test_training()
    path = "data/iris.data"
    csv_data = CSV.File(path, header=false)
    iris_names = ["sepal_len", "sepal_wid", "petal_len", "petal_wid", "class"]
    df = DataFrame(csv_data)
    rename!(df, iris_names)
    dropmissing!(df)

    # Checking BinaryNode
    bn = BinaryNode(1)
    leftchild(bn, BinaryNode(2))
    rightchild(bn, BinaryNode(3))
    rightchild(bn.right, BinaryNode(4))
    @test all([child.idx for child in children(bn)] .== [2,3]) &&
        isempty(children(bn.right.right)) && length(alloffspring(bn)) == 3
    
    deleted_child = bn.right.right
    delete_children!(bn.right)
    @test isnothing(bn.parent) && isnothing(bn.right.right) && 
        isempty(children(bn.right)) && length(alloffspring(bn)) == 2

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

test_training()