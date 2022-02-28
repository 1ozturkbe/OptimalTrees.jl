""" Loads Iris dataset for tests. """
function load_irisdata()
    path = "data/iris.data"
    csv_data = CSV.File(path, header=false)
    iris_names = ["sepal_len", "sepal_wid", "petal_len", "petal_wid", "class"]
    df = DataFrame(csv_data)
    rename!(df, iris_names)
    dropmissing!(df)
    return df
end

""" Turns classes in DataFrame (last column) into binary classes. """
function binarize(df = load_irisdata())
    df_bin = copy(df)
    X = df_bin[:,1:end-1]
    Y = df_bin[:, end]
    classes = unique(Y)
    for i = 1:length(Y)
        if Y[i] != classes[1]
            Y[i] = classes[2]
        end
    end
    df_bin[:, end] = Y
    return df_bin
end