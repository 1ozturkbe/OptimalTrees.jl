""" Tests the functionalities of BinaryNode. """
function test_binarynode()
    @info "Testing BinaryNode..."
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

""" Tests full MIO solution functionalities of MIOTree. """
function test_miotree()
    @info "Testing MIOTree..."
    d = MIOTree_defaults()
    d = MIOTree_defaults(:max_depth => 4, :cp => 1e-5)
    @test d[:max_depth] == 4
    @test get_param(d, :cp) == 1e-5
    mt = MIOTree(SOLVER_SILENT; max_depth = 4, minbucket = 0.03)
    df = load_irisdata()
    df = df[1:Int(floor(size(df, 1)/2)),:]
    X = Matrix(df[:,1:4])
    Y =  Array(df[:, "class"])
    md = 2
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
    @test OT.complexity(mt) == sum(as .!= 0)
    prune!(mt)
    @test check_if_trained(mt)
    @test length(allleaves(mt)) == sum(isapprox.(ckt, 1, atol = 1e-4)) == count(!isnothing(node.label) for node in allnodes(mt))
    @test all(getproperty.(apply(mt, X), :label) .== predict(mt, X))

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
    @info "Testing hyperplane CART..."
    mt = MIOTree(SOLVER_SILENT, max_depth = 5)
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

    # Checking deepening trees, and warmstarting
    deepen_to_max_depth!(mt)
    clean_model!(mt)
    generate_MIO_model(mt, X, Y)
    warmstart(mt)
    optimize!(mt)
    populate_nodes!(mt)
    prune!(mt)
        
    @test isapprox(score(mt, X, Y), 1, atol = 0.05)
    leaves = allleaves(mt)
    not_leaves = [nd for nd in allnodes(mt) if !is_leaf(nd)]
    @test all(is_leaf.(not_leaves) .== false)
    @test all(isnothing.(getproperty.(not_leaves, :a)) .== false)
    # @test all(isnothing.(getproperty.(leaves, :label)) .== false) # TODO: fix this test. 
end

function test_sequential()
    @info "Testing sequential training..."
    mt = MIOTree(SOLVER_SILENT; max_depth = 2, minbucket = 0.03)
    df = load_irisdata()
    X = Matrix(df[:,1:4])
    Y =  Array(df[:, "class"])
    clean_model!(mt)
    sequential_train!(mt, X, Y, pruning = true)
    @test true
end

function test_regression()
    @info "Testing regression..."
    feature_names = MLDatasets.BostonHousing.feature_names()
    n_samples = 20
    X_all = Matrix(transpose(MLDatasets.BostonHousing.features()))
    Y_all = Array(transpose(MLDatasets.BostonHousing.targets()))
    X = X_all[1:n_samples, :]
    Y = Y_all[1:n_samples, :]
    mt = MIOTree(SOLVER_SILENT, max_depth = 2, regression = true)
    generate_binary_tree(mt)
    generate_MIO_model(mt, X, Y)
    optimize!(mt)
    populate_nodes!(mt)
    prune!(mt)

    @test check_if_trained(mt)
    score1 = score(mt, X, Y)
    
    # Upping number of samples, and warmstarting
    n_samples = 30
    X = Matrix(transpose(MLDatasets.BostonHousing.features()))[1:n_samples, :]
    Y = Array(transpose(MLDatasets.BostonHousing.targets()))[1:n_samples, :]
    clean_model!(mt)
    generate_MIO_model(mt, X, Y)
    warmstart(mt)
    optimize!(mt)
    populate_nodes!(mt)
    prune!(mt)
    @test check_if_trained(mt)
    score2 = score(mt, X, Y)
    @test score1 <= 5 && score2 <= 5
end

function test_ensemblereg()
    @info "Testing ensemble regression..."
    feature_names = MLDatasets.BostonHousing.feature_names()
    Y = Array(transpose(MLDatasets.BostonHousing.targets()))
    # Shuffled data
    shuffle_idxs = shuffle(1:Int(length(Y)/2))
    Y = Y[shuffle_idxs]
    X = Matrix(transpose(MLDatasets.BostonHousing.features()))[shuffle_idxs, :]

    te = TreeEnsemble(SOLVER_SILENT; regression = true, max_depth = 1)
    plant_trees(te, 12)
    generate_binary_tree.(te.trees)
    train_ensemble(te, X, Y)
    populate_nodes!.(te.trees)
    prune!.(te.trees)
    @test all(check_if_trained.(te.trees))
    weigh_trees(te, X, Y)
    @test isapprox(sum(te.weights), 1, atol = 1e-3)
    @test score(te, X, Y) <= 7
end

test_binarynode()

test_miotree()

test_hyperplanecart()

test_sequential()

test_regression()

test_ensemblereg()


