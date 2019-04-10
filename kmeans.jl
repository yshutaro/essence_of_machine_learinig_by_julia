module kmeans

using Random

mutable struct KMeans
    n_clusters
    max_iter::Int
    random_seed
    labels_
    cluster_centers_
    function KMeans(n_clusters)
        new(n_clusters, 1000, 0, Nothing, Nothing)
    end
end

function fit(obj::KMeans, X)
    Random.seed!(obj.random_seed)

    labels_prev = zeros(size(X)[1])
    count = 0
    obj.cluster_centers_ = zeros(obj.n_clusters, size(X)[2])
    while (!(obj.labels_ .== labels_prev) && count < obj.max_iter)
        for i in 1:obj.n_clusters
            XX = X[obj.labels_ .== i, :]
        end
        # TODO
        dist = []
        labels_prev = obj.labels_
        obj.labels_ = argmin(dist)
        count += 1
    end
end

function predict(obj::KMeans, X)
    # TODO
    dist = []
    labels = argmin(dist)
end

end
