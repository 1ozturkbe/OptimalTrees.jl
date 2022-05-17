using Pkg
Pkg.activate("test")

using Clustering
using DataFrames
using GLPK
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
const SOLVER_SILENT = GLPK.Optimizer
include("utilities.jl")