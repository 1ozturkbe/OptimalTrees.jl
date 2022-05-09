using Pkg
Pkg.activate("test")

using Clustering
using DataFrames
using Gurobi
using JuMP
using CSV
using MathOptInterface
using MLDatasets
using Test
using Random

Random.seed!(1);
include("../src/OptimalTrees.jl")
using .OptimalTrees
global OT = OptimalTrees
SOLVER_SILENT = optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag" => 0)

include("utilities.jl")