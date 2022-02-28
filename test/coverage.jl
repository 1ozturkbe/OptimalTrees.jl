#=
test_coverage:
- Julia version: 
- Author: Berk
- Date: 2022-01-26
=#

# Run julia17 --project=test/. --code-coverage test/all.jl
# Then restart julia from within OptimalConstraintTree and run this file.
using Pkg
Pkg.activate("test")
global PROJECT_ROOT = @__DIR__
ENV["CODECOV_TOKEN"]= "dfef6217-c401-4f55-91b7-efbf1afc6ca0"
using Coverage
# process '*.cov' files
coverage = process_folder() # defaults to src/; alternatively, supply the folder name as argument
# process '*.info' files
coverage = merge_coverage_counts(coverage, filter!(
    let prefixes = (joinpath(pwd(), "src", ""))
        c -> any(p -> startswith(c.filename, p), prefixes)
    end,
    LCOV.readfolder("test")))
# Get total coverage for all Julia files
covered_lines, total_lines = get_summary(coverage)
# Or process a single file
@show get_summary(process_file(joinpath("src", "OptimalTrees.jl")))
println("Covered lines: ", covered_lines)
println("Total lines: ", total_lines)
println("Ratio: ", covered_lines/total_lines)

# Submit to coverage
Codecov.submit_local(coverage)