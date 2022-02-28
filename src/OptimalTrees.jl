module OptimalTrees
    using DataFrames
    using DocStringExtensions
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

    const MOI = MathOptInterface
    const OPTIMALTREES_ROOT = dirname(dirname(@__FILE__))
    const DATA_DIR = OPTIMALTREES_ROOT * "\\data\\"

    # Structs
    export MIOTree, BinaryNode, 

    # Tree building
        leftchild, rightchild, children, 
        alloffspring, printnode,
        generate_binary_tree, MIOTree_defaults, MIOTree,
        generate_MIO_model, delete_children!, prune!, chop_down!,
        populate_nodes!,
        apply, predict,
        is_leaf,
        get_classification_label, 
        set_classification_label!, 
        get_split_values, set_split_values!,

    # Tree training
        SVM, hyperplane_cart,

    # Scoring functions
        score, complexity,

    # Helper functions
        set_param, get_param,
        allnodes, allleaves
end