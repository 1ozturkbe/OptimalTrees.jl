using Pkg
Pkg.activate("test")

using DataFrames
using JuMP
using CPLEX
using CSV
using MathOptInterface
using Test
using Random

global MOI = MathOptInterface
Random.seed!(1);
MOI.Silent() = true;
include("../src/OptimalTrees.jl")
using .OptimalTrees
global OT = OptimalTrees
CPLEX_SILENT = OptimalTrees.CPLEX_SILENT