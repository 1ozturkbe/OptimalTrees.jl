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
function normalize(X)
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
function denormalize(X, bounds::Vector)
    X_denorm = zeros(size(X))
    for i = 1:size(X,2)
        X_denorm[:,i] = X[:,i] .* (bounds[i][2] - bounds[i][1]) .+ bounds[i][1]
    end
    return X_denorm
end

""" 
    $(TYPEDSIGNATURES)

Denormalizes a regressor using provided bounds. 
"""
function denormalize_regressor(β0::Real, β::Vector, X_max::Vector, X_min::Vector, Y_max::Real, Y_min::Real)
    β_denorm = β ./ (X_max - X_min) * (Y_max - Y_min)
    β0_denorm = (sum(-β.* (X_min ./ (X_max - X_min))) + β0) * (Y_max - Y_min) + Y_min  
    return β0_denorm, β_denorm
end