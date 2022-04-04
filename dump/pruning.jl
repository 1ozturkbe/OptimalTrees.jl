""" Fixes the unlabeled leaves of an MIOTree.  """
function fix_labeling!(mt::MIOTree)
    queue = allleaves(mt)
    while !isempty(queue)    
        leaf = popfirst!(queue)      
        if isnothing(leaf.label) && !isnothing(leaf.parent)
            par = leaf.parent
            labels = [child.label for child in children(leaf.parent) if !isnothing(child.label)]
            if length(labels) == 1
                set_classification_label!(par, labels[1])
                delete_children!(par)
                if !isnothing(par.parent)
                    append!(queue, children(par.parent))
                end
            end
        end
    end
    return
end