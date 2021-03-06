""" Tests the functionalities of BinaryNode. """
function test_binarynode()
    @info "Testing BinaryNode..."
    bn = BinaryNode(1)
    @test_throws ErrorException get_parent(bn)
    leftchild(bn, BinaryNode(2))
    rightchild(bn, BinaryNode(3))
    rightchild(bn.right, BinaryNode(4))
    @test all([child.idx for child in children(bn)] .== [2,3]) &&
        isempty(children(bn.right.right)) && length(alloffspring(bn)) == 3
    
    delete_children!(bn.right)
    @test_throws ErrorException get_lower_child(bn.right)
    @test_throws ErrorException get_upper_child(bn.right)
    @test isnothing(bn.parent) && isnothing(bn.right.right) && 
        isempty(children(bn.right)) && length(alloffspring(bn)) == 2

    @test_throws ErrorException set_split_values!(bn.right, [1,2,3], 4)
    @test_throws ErrorException set_classification_label!(bn, 5)
end

function test_data_processing()
    @info "Testing data processing..."
    Y = transpose(MLDatasets.BostonHousing.targets())
    X = transpose(MLDatasets.BostonHousing.features())

    @test_throws ErrorException split_data(X, Y, bins = 3, sample_proportion = [0.1, 0.2, 0.7])
    @test_throws ErrorException split_data(X, Y, sample_proportion = [0.1, 0.2, 0.3])
    @test_throws ErrorException split_data(X, Y, sample_count = [10, 20, 30])

    data = split_data(X, Y)
    @test length(data) == 2 && length(data[1][2]) + length(data[2][2]) == length(Y)
    data = split_data(X, Y, bins = 9)
    @test length(data) == 9 && sum(length(dat[2]) for dat in data) == length(Y) && all(isapprox(length(dat[2]), length(Y)/9, atol = 1) for dat in data)
    data = split_data(X, Y, sample_count = [100, 200, 206])
    @test length(data) == 3 && sum(length(dat[2]) for dat in data) == length(Y) 
end

""" Tests full MIO solution functionalities of MIOTree. """
function test_miotree()
    @info "Testing MIOTree..."
    d = MIOTree_defaults()
    d = MIOTree_defaults(max_depth =  4, cp = 1e-5)
    @test d[:max_depth] == 4
    @test get_param(d, :cp) == 1e-5
    mt = MIOTree(SOLVER_SILENT; max_depth = 4, minbucket = 0.03)
    df = load_irisdata()
    df = df[1:Int(floor(size(df, 1)/2)),:]
    X = Matrix(df[:,1:4])
    Y =  Array(df[:, "class"])
    md = 2
    set_params!(mt, max_depth = md, minbucket = 0.001)
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

    # Testing cloning
    new_mt = clone(mt)
    @test all(get_split_values(mt.root) .== (get_split_weights(new_mt.root), get_split_threshold(new_mt.root)))

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
    fit!(mt, "cart", X, Y)
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
    fit!(mt, "mio", X, Y)
    @test all(label isa Tuple for label in get_classification_label.(allleaves(mt)))
    @test all(label isa Real for label in get_regression_constant.(allleaves(mt)))
    @test all(label isa AbstractArray for label in get_regression_weights.(allleaves(mt)))

    @test check_if_trained(mt)
    score1 = score(mt, X_all, Y_all)
    
    # Upping number of samples, and warmstart procedure
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
    score2 = score(mt, X_all, Y_all)
    @test score1 <= 1 && score2 <= 1
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
    plant_trees(te, 10)
    fit!(te, "mio", X, Y)
    @test all(check_if_trained.(te.trees))
    weigh_trees(te, X, Y)
    @test isapprox(sum(te.weights), 1, atol = 1e-5)
    @test score(te, X, Y) >= 0.5
end

function test_ensemblecls()
    @info "Testing ensemble classification... "
    feature_names = MLDatasets.BostonHousing.feature_names()
    Y = Array(transpose(MLDatasets.BostonHousing.targets()))
    shuffle_idxs = shuffle(1:Int(length(Y)))
    Y = Array(Y[shuffle_idxs] .>= 20)
    X = Matrix(transpose(MLDatasets.BostonHousing.features()))[shuffle_idxs, :]
    te = TreeEnsemble(SOLVER_SILENT; max_depth = 3)
    plant_trees(te, 11)
    fit!(te, "cart", X, Y)
    @test all(check_if_trained.(te.trees))
    @test score(te, X, Y) >= 0.82
end

function test_cluster_heuristic()
    @info "Testing clustering heuristic in classification..."
    Y = Array(transpose(MLDatasets.BostonHousing.targets()))
    shuffle_idxs = shuffle(1:Int(length(Y)))
    Y = Array(Y[shuffle_idxs])
    X = Matrix(transpose(MLDatasets.BostonHousing.features()))[shuffle_idxs, :]

    X_norm, X_bounds = normalize(X)
    Y_norm, Y_bounds = normalize(Y)
    dists = pairwise_distances(X_norm)
    clust = hclust(dists)
    max_depth = 5
    n_clust = 2^(max_depth-1)
    idxs = cutree(clust; k = n_clust, h = max_depth)
    cluster_bins = Dict(i => [] for i in unique(idxs))
    [push!(cluster_bins[idxs[i]], i) for i = 1:length(idxs)]
    subset_idxs = []
    sample_proportion = 0.125
    for (key, val) in cluster_bins # Picking just one-eighth of all samples in clusters
        append!(subset_idxs, val[1:Int(ceil(sample_proportion*length(val)))])
    end
    @test sample_proportion <= length(subset_idxs)/length(Y) <= 2*sample_proportion

    mt = MIOTree(SOLVER_SILENT, max_depth = max_depth)
    generate_binary_tree(mt)
    generate_MIO_model(mt, X_norm[subset_idxs, :], Array(Y_norm .>= 0.3)[subset_idxs])
    optimize!(mt)
    populate_nodes!(mt)
    prune!(mt)
    @test score(mt, X_norm, Array(Y_norm .>= 0.3)) >= 0.8
end

test_binarynode()

test_data_processing()

test_miotree()

test_hyperplanecart()

test_sequential()

test_regression()

test_ensemblereg()

test_ensemblecls()

test_cluster_heuristic()