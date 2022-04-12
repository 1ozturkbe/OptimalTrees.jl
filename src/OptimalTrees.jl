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
    include("tree_utils.jl")
    include("training.jl")

    const MOI = MathOptInterface
    const OPTIMALTREES_ROOT = dirname(dirname(@__FILE__))
    const DATA_DIR = OPTIMALTREES_ROOT * "\\data\\"

    # Structs
    export MIOTree, BinaryNode, 

    # Tree building
        MIOTree_defaults, MIOTree,
        leftchild, rightchild, children, 
        printnode, generate_binary_tree, 
        delete_children!, 
        populate_nodes!, prune!, 
        chop_down!,
        
    # More advanced tree building
        deepen_to_max_depth!, 
        deepen_one_level!, 

    # Tree querying
        is_leaf, depth, lineage, alloffspring,
        get_classification_label, 
        set_classification_label!, 
        get_split_values, set_split_values!,
        apply, predict,

    # Tree training
        SVM, ridge_regress, 
        hyperplane_cart,
        warmstart, generate_MIO_model, 
        sequential_train!,

    # Tree utilities
        find_leaves, check_if_trained, 
        trust_region_data, pwl_constraint_data,

    # Scoring functions
        score,

    # Helper functions
        set_param, get_param,
        normalize,
        allnodes, allleaves,
        clean_model!,

    # Debugging
        debug_if_trained
end