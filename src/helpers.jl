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

""" gets parameters within Dict. """
function get_param(gm::Dict, key::Symbol)
    if haskey(gm, key)
        return gm[key]
    else
        throw(ErrorException("Parameter with key " * string(key) * " is invalid."))
    end
end