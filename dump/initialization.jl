# set_param!(mt, :max_depth, 1)
# generate_MIO_model(mt, X, Y)
# set_optimizer(mt, SOLVER_SILENT)
# optimize!(mt)
# for i = 2:get_param(mt, :max_depth)
#     set_param!(mt, :max_depth, i)
#     generate_MIO_model(mt, X, Y)
#     set_optimizer(mt, SOLVER_SILENT)
#     optimize!(mt)
# end