""" Sets parameters within Dict. """
function set_param(gm::Dict, key::Symbol, val, checks = true)
    if haskey(gm, key) && val isa typeof(gm[key])
        gm[key] = val
        return
    elseif checks
        throw(ErrorException("Parameter with key " * string(key) * " is invalid."))
    else 
        return
    end
end

""" 
    $(TYPEDSIGNATURES)

Gets parameters within Dict. 
"""
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

""" 
    $(TYPEDSIGNATURES)

De-normalizes a given chunk of data by column. 
"""
function denormalize(X::Matrix, bounds::Vector)
    X_denorm = zeros(size(X))
    for i = 1:size(X,2)
        X_denorm[:,i] = X[:,i] .* (bounds[i][2] - bounds[i][1]) .+ bounds[i][1]
    end
    return X_denorm
end