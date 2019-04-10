module kmeans

using Random

mutable struct KMeans
    n_clusters
    max_iter::Int
    random_seed
    cluster_centers_
    function KMeans(n_clusters)
        new(n_clusters, 1000, 0, Nothing)
    end
end

function fit(obj::KMeans, X)
    Random.seed!(obj.random_seed)

    count = 0
    obj.cluster_centers_ = zeros(obj.n_clusters, size(X)[2])
end

end
