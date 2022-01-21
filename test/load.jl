
using DataFrames
using JuMP
using CPLEX
using CSV
using MathOptInterface
using Test
using Random

global MOI = MathOptInterface
global CPLEX_SILENT = with_optimizer(CPLEX.Optimizer, CPX_PARAM_SCRIND = 0)
Random.seed!(1);
MOI.Silent() = true;
include("../src/OptimalTrees.jl")
using .OptimalTrees
global OT = OptimalTrees