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