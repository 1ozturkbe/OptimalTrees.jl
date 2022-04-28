""" Sets parameters within Dict. """
function set_param!(d::Dict, key::Symbol, val, checks = true)
    if haskey(d, key) && val isa typeof(d[key])
        d[key] = val
        return
    elseif checks
        throw(ErrorException("Parameter with key " * string(key) * " is invalid."))
    else 
        return
    end
end

function set_params!(d::Dict; kwargs...)
    for (k,v) in kwargs
        set_param!(d, k, v)
    end
    return
end

""" Gets parameters within Dict. """
function get_param(gm::Dict, key::Symbol)
    if haskey(gm, key)
        return gm[key]
    else
        throw(ErrorException("Parameter with key " * string(key) * " is invalid."))
    end
end

""" 
    $(TYPEDSIGNATURES)

Normalizes a given chunk of data by column. 
"""
function normalize(X::Matrix)
    bounds = [(minimum(X[:,i]), maximum(X[:,i])) for i = 1:size(X, 2)]
    X_norm = zeros(size(X))
    for i = 1:size(X,2)
        X_norm[:,i] = (X[:,i] .- bounds[i][1]) ./ (bounds[i][2] - bounds[i][1])
    end
    return X_norm, bounds
end

function normalize(X::Vector)
    bounds = (minimum(X), maximum(X))
    X_norm = (X .- bounds[1]) ./ (bounds[2] - bounds[1])
    return X_norm, bounds
end

""" 
    $(TYPEDSIGNATURES)

Returns the distance matrix of X. Should almost always be done on a normalized X matrix. 
"""
function pairwise_distances(X_norm::Matrix)
    dists = zeros(size(X_norm, 1), size(X_norm, 1))
    for i = 1:size(X_norm, 1)
        for j = i:size(X_norm, 1)
            dists[i,j] = sum((X_norm[i,:] .- X_norm[j,:]).^2)
            dists[j, i] = dists[i,j]
        end
    end
    return dists
end

""" Computes the mode, i.e. the most common element of an array. """
function mode(X::Vector)
    count_dict = Dict()
    for i = 1:length(X)
        if !(X[i] in keys(count_dict))
            count_dict[X[i]] = 1
        else
            count_dict[X[i]] += 1
        end
    end
    maxval = maximum(values(count_dict))
    return [k for (k,v) in count_dict if v == maxval]
end

"""
    $(TYPEDSIGNATURES)

Splits data according to the kwargs. Potential kwargs:
- bins::Int: The number of bins that you would like the data to fit into. 
- sample_proportion::Union{Array, Real}: Proportion of samples in each bin. If Real, means only two bins. 
- sample_count::Union{Array}: Number of samples in each bin. 
"""
function split_data(X, Y; 
    bins::Union{Int, Nothing} = nothing,
    sample_proportion::Union{Array, Real, Nothing} = nothing,
    sample_count::Union{Array, Nothing} = nothing)

    (sum(!isnothing(bins) + !isnothing(sample_proportion) + !isnothing(sample_count)) <= 1) || throw(ErrorException("split_data only takes one or zero kwargs."))

    subset_idxs = [0]
    if sum(!isnothing(bins) + !isnothing(sample_proportion) + !isnothing(sample_count)) == 0
        sample_proportion = [0.5, 0.5]
    elseif !isnothing(bins)
        sample_proportion = [1/bins for i = 1:bins]
    end

    if !isnothing(sample_proportion)
        isapprox(sum(sample_proportion), 1, atol = 1e-7) || throw(ErrorException("Sum of sample-proportion in split_data must be equal to the number of samples. "))
        for i = 1:length(sample_proportion)
            append!(subset_idxs, Int(ceil(length(Y)*sum(sample_proportion[1:i]))))
        end
    elseif !isnothing(sample_count)
        sum(sample_count) == size(X, 1) || throw(ErrorException("Sum of sample-count in split_data must be equal to the number of samples. "))
        for i = 1:length(sample_count)
            push!(subset_idxs, subset_idxs[end] + sample_count[i])
        end        
    end
    return [(X[subset_idxs[i]+1:subset_idxs[i+1], :], 
    Y[subset_idxs[i]+1:subset_idxs[i+1]]) for i = 1:length(subset_idxs)-1]
end