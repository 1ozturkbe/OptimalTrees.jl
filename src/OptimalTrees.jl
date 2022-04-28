module OptimalTrees
    using Clustering
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
    include("ensemble.jl")

    const MOI = MathOptInterface
    const OPTIMALTREES_ROOT = dirname(dirname(@__FILE__))
    const DATA_DIR = OPTIMALTREES_ROOT * "\\data\\"

    # Structs
    export MIOTree, BinaryNode, TreeEnsemble,

    # Tree building
        MIOTree_defaults,
        leftchild, rightchild, children, 
        get_lower_child, get_upper_child, get_parent,
        printnode, generate_binary_tree, 
        delete_children!, 
        populate_nodes!, prune!, 
        chop_down!,
        
    # More advanced tree building
        deepen_to_max_depth!, 
        deepen_one_level!, clone, 

    # Tree querying
        is_leaf, depth, 
        lineage, alloffspring,
        get_classification_label, 
        set_classification_label!, 
        get_split_values, set_split_values!,
        get_split_threshold, get_split_weights,
        get_regression_constant, get_regression_weights,
        apply, predict,

    # Tree training
        fit!, 
        SVM, hyperplane_cart,
        warmstart, generate_MIO_model, 
        sequential_train!,

    # Tree utilities
        find_leaves, check_if_trained, 
        trust_region_data, pwl_constraint_data,

    # Scoring functions
        score,

    # Helper functions
        set_param!, set_params!, get_param,
        normalize, pairwise_distances, mode,
        allnodes, allleaves,
        clean_model!,
        split_data,

    # Tree ensembles
        TreeEnsemble_defaults,
        plant_trees, weigh_trees,

    # Debugging
        debug_if_trained
end