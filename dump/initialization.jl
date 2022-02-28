""" Initializes guesses for MIOTree based on previous solution. """
function initialize_guesses(mt::MIOTree)
    m = mt.model
    for nd in allnodes(mt)
        if is_leaf(nd) && !isnothing(nd.label)
            for i = 1:length(mt.classes)
                if mt.classes[i] == nd.label
                    JuMP.set_start_value(m[:ckt][i, nd.idx], 1)
                else
                    JuMP.set_start_value(m[:ckt][i, nd.idx], 0)
                end
            end
        elseif !isnothing(nd.a)
            JuMP.set_start_value.(m[:a][nd.idx, :], nd.a)
            JuMP.set_start_value(m[:b][nd.idx], nd.b)
        end
    end
    return
end

# set_param(mt, :max_depth, 1)
# generate_MIO_model(mt, X, Y)
# set_optimizer(mt, SOLVER_SILENT)
# optimize!(mt)
# for i = 2:get_param(mt, :max_depth)
#     set_param(mt, :max_depth, i)
#     generate_MIO_model(mt, X, Y)
#     set_optimizer(mt, SOLVER_SILENT)
#     optimize!(mt)
# end