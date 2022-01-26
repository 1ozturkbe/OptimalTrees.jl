include("load.jl");

@testset "OptimalTrees.jl" begin
    include(string(OptimalTrees.OPTIMALTREES_ROOT, "/test/src.jl"))
end