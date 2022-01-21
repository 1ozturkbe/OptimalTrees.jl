module OptimalTrees
    using DataFrames
    using JuMP
    using MathOptInterface
    using Missings
    using Parameters
    using ProgressMeter
    using Random

    include("helpers.jl")
    include("binarynode.jl")
    include("tree.jl")
    include("training.jl")

    const OPTIMALTREES_ROOT = dirname(dirname(@__FILE__))
    const DATA_DIR = OPTIMALTREES_ROOT * "\\data\\"

    # Structs
    export MIOTree, BinaryNode, 

    # Tree building
        leftchild, rightchild, children, 
        alloffspring, printnode,
        generate_binary_tree, MIOTree_defaults, MIOTree, build_MIOTree,
        generate_tree_model, delete_children!, prune!, 
        populate_nodes!,
        apply, score, complexity,
        is_leaf, get_classification_label, 
        get_split_weights, get_split_threshold,

    # Helper functions
        set_param, get_param
end