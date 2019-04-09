module kmeans

mutable struct KMeans
    n_clusters
    max_iter::Int
    random_seed
    function KMeans(n_clusters)
        new(n_clusters, 1000, 0)
    end
end

function fit()
    
end

end
