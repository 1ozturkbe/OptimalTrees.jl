JuMP.set_optimizer(mt::MIOTree, solver) = JuMP.set_optimizer(mt.model, solver)

JuMP.optimize!(mt::MIOTree) = JuMP.optimize!(mt.model)

"""
    generate_tree_model(mt::MIOTree, X::Matrix, Y::Array)

Generates a JuMP MIO model of the tree classifier, with data X and Y.
"""
function generate_tree_model(mt::MIOTree, X::Matrix, Y::Array)
    if !isempty(mt.model.obj_dict)
        mt.model = JuMP.Model() # TODO: find a way to add solver in here.
    end
    n_samples, n_vars = size(X)

    # Reference minimal parameters
    max_depth = get_param(mt, :max_depth)
    n_nodes = length(mt.nodes)
    n_leaves = length(mt.leaves)
    n_splits = n_nodes - n_leaves
    nd_idxs = [nd.idx for nd in mt.nodes] # Node indices
    lf_idxs = [lf.idx for lf in mt.leaves] # Leaf indices
    sp_idx = [idx for idx in nd_idxs if idx ∉ lf_idxs]
    min_points = get_param(mt, :minbucket)
    if !isa(min_points, Int) && 0 <= min_points <= 1
        min_points = Int(ceil(min_points * n_samples))
    else
        throw(ErrorException("Minbucket parameter must be between 0-1 or an integer!"))
    end
    mt.classes = sort(unique(Y)) # The potential classes are sorted. 
    k = length(mt.classes)
    k != 2 && @info("Detected $(k) unique classes.")

    @variable(mt.model, -1 <= a[1:n_splits, 1:n_vars] <= 1)
    @variable(mt.model, 0 <= abar[1:n_splits, 1:n_vars])
    @variable(mt.model, -1 <= b[1:n_splits] <= 1)
    @variable(mt.model, d[1:n_splits], Bin)
    @variable(mt.model, s[1:n_splits, 1:n_vars], Bin) # Binary variables for complexity penalty
    @constraint(mt.model, -b .<= d)
    @constraint(mt.model, b .<= d)
    @constraint(mt.model, -a .<= s)
    @constraint(mt.model, a .<= s)
    @constraint(mt.model, a .<= abar)
    @constraint(mt.model, -a .<= abar)
    @constraint(mt.model, [j=1:n_vars], s[:,j] .<= d[:])
    @constraint(mt.model, [i=1:n_splits], sum(s[i, :]) >= d[i])
    @constraint(mt.model, [i=1:n_splits, j=1:n_vars], a[i,j] <= s[i,j])
    @constraint(mt.model, [i=1:n_splits, j=1:n_vars], -s[i,j] <= a[i,j])
    
    # Enforcing each point to one leaf
    @variable(mt.model, z[1:n_samples, 1:n_leaves], Bin)
    @constraint(mt.model, [i=1:n_samples], sum(z[i, :]) == 1)

    # Making sure that variables are properly binned. 
    @variable(mt.model, ckt[1:k, 1:n_leaves], Bin)  # Class at leaf
    @variable(mt.model, Nt[1:n_leaves] >= 0)       # Total number of points at leaf
    @variable(mt.model, lt[1:n_leaves], Bin)       # Whether or not a leaf is occupied
    @variable(mt.model, Nkt[1:k, 1:n_leaves] >= 0) # Number of points of at leaf with class k

    @constraint(mt.model, [i = 1:n_leaves], Nt[i] == sum(z[:, i])) # Counting number of points in a leaf. 
    @constraint(mt.model, [i = 1:n_leaves], sum(ckt[:, i]) == lt[i]) # Making sure a class is only assigned if leaf is occupied.
    for kn = 1:k
        # Number of values of each class
        @constraint(mt.model, [i=1:n_leaves], Nkt[kn, i] == 
                    sum(z[l, i] for l = 1:n_samples if Y[l] == mt.classes[kn]))
    end

    # Loss function
    @variable(mt.model, Lt[1:n_leaves] >= 0)
    @constraint(mt.model, [i = 1:n_leaves, j = 1:k], Lt[i] >= Nt[i] - Nkt[j, i] - n_samples * (1-ckt[j,i]))
    @constraint(mt.model, [i = 1:n_leaves, j = 1:k], Lt[i] <= Nt[i] - Nkt[j, i] + n_samples * ckt[j,i])

    @constraint(mt.model, sum(abar[1, :]) <= d[1])
    for nd in mt.nodes
        if !is_leaf(nd) && !isnothing(nd.parent)
            @constraint(mt.model, d[nd.idx] <= d[nd.parent.idx])
            @constraint(mt.model, sum(abar[nd.idx, :]) <= d[nd.idx])
        end
    end

    mu = 5e-5
    for j = 1:n_leaves
        lf = mt.leaves[j]
        # Enforcing minbucket 
        @constraint(mt.model, [i=1:n_samples], z[i, j] <= lt[j])
        @constraint(mt.model, sum(z[:, j]) >= min_points*lt[j])
        # Enforcing hyperplane splits
        for i=1:n_samples
            nd = lf
            for _ = 1:max_depth # while loop wasn't working... scoping is hard
                if nd.idx == nd.parent.left.idx
                    @constraint(mt.model, sum(a[nd.parent.idx, :] .* X[i, :]) + mu <= b[nd.parent.idx] + (2+mu)*(1-z[i,j])) 
                elseif nd.idx == nd.parent.right.idx
                    @constraint(mt.model, sum(a[nd.parent.idx, :] .* X[i, :]) >= b[nd.parent.idx] - 2*(1-z[i,j])) 
                else
                    throw(ErrorException("Node backtracking failed for some reason."))
                end
                try 
                    nd = nd.parent
                catch err
                    if isa(err, UndefRefError)
                        break
                    else
                        throw(ErrorException("Node parenting failed unexpectedly."))
                        break
                    end
                end
            end
        end
    end

    # Objective function: misclassification error + complexity
    @objective(mt.model, Min, 1/n_samples * sum(Lt) + get_param(mt, :cp) * sum(s[:,:]))
    return
end