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
function test_miotree()
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
    set_param(mt, :minbucket, 0.001)
    generate_binary_tree(mt)
    generate_MIO_model(mt, X, Y)
    @test length(allleaves(mt)) == 2^md

    set_optimizer(mt, SOLVER_SILENT)
    optimize!(mt)
    @test !check_if_trained(mt) # Trees must be populated and pruned before they qualify!

    m = mt.model
    as, d, Lt, ckt, Nkt = [getvalue.(m[i]) for i in [:a, :d, :Lt, :ckt, :Nkt]]; 
    populate_nodes!(mt)
    @test length(allleaves(mt)) == 2^md
    @test sum(!isnothing(node.a) for node in allnodes(mt)) == 2^md-1 - sum(all(isapprox.(as[i,:], 0, atol = 1e-10)) for i = 1:2^md-1)
    @test sum(!isnothing(node.label) for node in allnodes(mt)) == length(Nkt) - sum(isapprox.(Nkt, 0, atol = 1e-5))
    prune!(mt)
    @test check_if_trained(mt)
    @test length(allleaves(mt)) == sum(isapprox.(ckt, 1, atol = 1e-4)) == prod(size(Nkt)) - sum(isapprox.(Nkt, 0, atol = 1e-4)) == count(!isnothing(node.label) for node in allnodes(mt))
    @test isapprox(score(mt, X, Y), 1-sum(Lt), atol=1e-1) && complexity(mt) == sum(as .!= 0)

    # # Plotting results for debugging
    # using Plots
    # colors = ["green", "red", "blue"]
    # labels = unique(Y)
    # plt = plot()
    # for i = 1:length(labels)
    #     label = labels[i]
    #     idxs = findall(x -> x == label, Y)
    #     plt = scatter!(X[idxs,1], X[idxs, 2], color = colors[i], label = labels[i])
    # end
    # display(plt)

    # # Plotting correct predictions
    # colors = ["purple", "yellow"]
    # labels = [0, 1]
    # plt = plot()
    # for i=1:length(labels)
    #     label = labels[i]
    #     idxs = findall(x -> x == label, predict(mt, X) .== Y)
    #     plt = scatter!(X[idxs,1], X[idxs, 2], color = colors[i], label = labels[i])
    # end
    # display(plt)

    # Checking data extraction
    ud, ld = trust_region_data(mt)
    @test all(sum(length(ud[lf.idx]) + length(ld[lf.idx])) == depth(lf) for lf in allleaves(mt)) # The right number of splits
end

function test_hyperplanecart()
    mt = MIOTree(SOLVER_SILENT)
    df = binarize(load_irisdata())
    X = Matrix(df[:,1:4])
    Y =  Array(df[:, "class"])
    hyperplane_cart(mt, X, Y)
    @test check_if_trained(mt)
    @test score(mt, X, Y) == 1.
    nds = apply(mt, X)
    @test all(getproperty.(nds, :label) .== Y)
    @test all(isnothing.(getproperty.(allleaves(mt), :label)) .== false)
    leaves = allleaves(mt)
    not_leaves = [nd for nd in allnodes(mt) if !is_leaf(nd)]
    @test all(is_leaf.(not_leaves) .== false)
    @test all(isnothing.(getproperty.(not_leaves, :a)) .== false)
    @test all(isnothing.(getproperty.(leaves, :label)) .== false)

    # Check that pruning does nothing
    prune!(mt)
    @test check_if_trained(mt)

    # Checking data extraction
    ud, ld = trust_region_data(mt)
    @test all(sum(length(ud[lf.idx]) + length(ld[lf.idx])) == depth(lf) for lf in allleaves(mt)) # The right number of splits
end

test_binarynode()

test_miotree()

test_hyperplanecart()