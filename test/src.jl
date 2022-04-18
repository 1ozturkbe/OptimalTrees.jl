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

function test_data()
    @info "Testing data-manipulating functions..."
    df = load_irisdata()
    X = Matrix(df[:,1:4])
    Y =  Array(df[:, "class"])
    X_norm, bounds = normalize(X)
    @test all(X_norm .>= 0) && all(X_norm .<= 1)
    X_denorm = denormalize(X_norm, bounds)
    @test all(isapprox.(X, X_denorm))

    # Testing regression denormalization
    X = 5*rand(100,3)
    base_b = [1, -1/2, 4]
    base_b0 = 3
    Y = X * base_b .+ base_b0
    X_norm, X_bounds = normalize(X)
    X_max = [maximum(bd) for bd in X_bounds]
    X_min = [minimum(bd) for bd in X_bounds]
    Y_norm, Y_bounds = normalize(Y)
    Y_max = maximum(Y_bounds[1])
    Y_min = minimum(Y_bounds[1])
    b0_norm, b_norm = ridge_regress(X_norm, Y_norm, mt.solver)
    b0, b = denormalize_regressor(b0_norm, b_norm, X_max, X_min, Y_max, Y_min)
    @test isapprox(b0, base_b0, atol=1e-2)
    @test all(isapprox.(b, base_b, atol = 1e-2))
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

function test_seqcls()
    @info "Testing sequential classification..."
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
    @test score(mt, X, Y) <= 1
    
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
    @test score(mt, X, Y) <= 1
end

# test_binarynode()

# test_data()

# test_miotree()

# test_hyperplanecart()

# test_seqcls()

# test_regression()

# mt = test_greedyreg()

# """ Finds the best split orthogonal to (β0, β) to minimize regression error. """
# function find_orthogonal_split(X, Y, β0, β)
#     mtos = MIOTree(SOLVER_SILENT, max_depth = 1)
#     generate_binary_tree(mtos)
#     generate_MIO_model(mtos, X, Y)


@info "Testing greedy regression..."
feature_names = MLDatasets.BostonHousing.feature_names()
n_samples = 30
X = Matrix(transpose(MLDatasets.BostonHousing.features()))
Y = Array(transpose(MLDatasets.BostonHousing.targets()))
X_norm, X_bounds = normalize(X)
X_max = [maximum(bd) for bd in X_bounds]
X_min = [minimum(bd) for bd in X_bounds]
Y_norm, Y_bounds = normalize(Y)
Y_max = maximum(Y_bounds[1])
Y_min = minimum(Y_bounds[1])

# mt = MIOTree(SOLVER_SILENT, max_depth = 1, regression = true)
# generate_binary_tree(mt)
# generate_MIO_model(mt, X, Y)
# optimize!(mt)

mt = MIOTree(SOLVER_SILENT)
set_param(mt, :regression, true)

n_samples, n_vars = size(X)
minpoints = ceil(n_samples * get_param(mt, :minbucket))
max_depth = get_param(mt, :max_depth)
regrtol = get_param(mt, :regrtol)
valid_leaves = [mt.root] # Stores leaves ready for SVM cuts. 
ct = 1
point_idxs = Dict(mt.root.idx => collect(1:n_samples))
β0, β = ridge_regress(X_norm, Y_norm, mt.solver)
β0, β = denormalize_regressor(β0, β, X_max, X_min, Y_max, Y_min)
set_classification_label!(mt.root, (β0, β))

while !isempty(valid_leaves)
    global ct
    leaf = popfirst!(valid_leaves)
    (β0, β) = get_classification_label(leaf)
    errors = Y[point_idxs[leaf.idx]] .- 
        (X[point_idxs[leaf.idx],:] * β .+ β0)
    if all(abs.(errors) .<= regrtol)
        continue
    end
    split_errors = errors .>= 0

    left_idxs = findall(x -> x <= 0, errors)
    right_idxs = findall(x -> x > 0, errors)

    # Optimal split tree
    # mtos = MIOTree(Gurobi.Optimizer, max_depth = 1)
    # generate_binary_tree(mtos)
    # generate_MIO_model(mtos, X_norm[point_idxs[leaf.idx],:], Y_norm[point_idxs[leaf.idx]])
    # @constraint(mtos.model, sum(β[i] * mtos.model[:a][1,i] for i=1:length(β))  == 0) # Orthogonality constraint
    # optimize!(mtos)
    # populate_nodes!(mtos)



    # a, b = SVM(X_norm[point_idxs[leaf.idx], :],
    #     Array(split_errors[point_idxs[leaf.idx]]), mt.solver)
    # left_idxs = findall(x -> x <= 0, 
    #     [sum(a .*X_norm[i, :]) - b for i = point_idxs[leaf.idx]])
    # right_idxs = findall(x -> x > 0, 
    #     [sum(a .*X_norm[i, :]) - b for i = point_idxs[leaf.idx]])

    if length(left_idxs) < minpoints || length(right_idxs) < minpoints
        continue
    end
    
    # Adding children, and setting split value. 
    ct += 1
    leftchild(leaf, BinaryNode(ct))
    ct += 1
    rightchild(leaf, BinaryNode(ct))
    set_split_values!(leaf, β, -β0)

    # Checking and labeling left child, and adding to queue
    point_idxs[leaf.left.idx] = point_idxs[leaf.idx][left_idxs]
    β0, β = ridge_regress(X_norm[point_idxs[leaf.left.idx],:], 
        Y_norm[point_idxs[leaf.left.idx]], mt.solver)
    β0, β = denormalize_regressor(β0, β, X_max, X_min, Y_max, Y_min)
    set_classification_label!(leaf.left, (β0, β))

    # Checking and labeling right child
    point_idxs[leaf.right.idx] =  point_idxs[leaf.idx][right_idxs]
    β0, β = ridge_regress(X_norm[point_idxs[leaf.right.idx],:], 
        Y_norm[point_idxs[leaf.right.idx]], mt.solver)
    β0, β = denormalize_regressor(β0, β, X_max, X_min, Y_max, Y_min)
    set_classification_label!(leaf.right, (β0, β))
    
    # Pruning if necessary, 
    # and choosing whether leaves should be added to queue. 
    if depth(leaf.left) < max_depth
        push!(valid_leaves, leaf.left)
    end
    if depth(leaf.right) < max_depth
        push!(valid_leaves, leaf.right)
    end
end

df = DataFrame(pred = predict(mt, X), act = vec(Y)) 
