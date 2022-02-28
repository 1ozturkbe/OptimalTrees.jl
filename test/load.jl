using Pkg
Pkg.activate("test")

using DataFrames
using JuMP
using CPLEX
using CSV
using MathOptInterface
using Test
using Random

Random.seed!(1);
include("../src/OptimalTrees.jl")
using .OptimalTrees
global OT = OptimalTrees
SOLVER_SILENT = OptimalTrees.SOLVER_SILENT