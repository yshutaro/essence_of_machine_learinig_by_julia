module kmeans

using Random
using Statistics
using Base.Iterators

mutable struct KMeans
    n_clusters
    max_iter::Int
    rng_::MersenneTwister
    labels_
    cluster_centers_
    function KMeans(n_clusters)
        rng = MersenneTwister(0)
        new(n_clusters, 1000, rng, Nothing, Nothing)
    end
end

function fit(obj::KMeans, X)
    cycle = Iterators.cycle(collect(1:obj.n_clusters))
    obj.labels_ = collect(Iterators.take(cycle,size(X)[1]))
    shuffle!(obj.rng_, obj.labels_)
    labels_prev = zeros(size(X)[1])
    count = 0
    obj.cluster_centers_ = zeros(obj.n_clusters, size(X)[2])
    while (!all(obj.labels_ .== labels_prev) && count < obj.max_iter)
        for i in 1:obj.n_clusters
            XX = X[obj.labels_ .== i, :]
            obj.cluster_centers_[i, :] = mean(XX, dims=1)
        end
        dist = [sum((X[i, :] .- obj.cluster_centers_[j, :]).^2)
         for i in 1:size(X)[1], j in 1:size(obj.cluster_centers_)[1]]
        labels_prev = obj.labels_
        obj.labels_ = vec([x[2] for x in argmin(dist, dims=2)])
        count += 1
    end
end

function predict(obj::KMeans, X)
    dist = [sum((X[i, :] .- obj.cluster_centers_[j, :]).^2)
     for i in 1:size(X)[1], j in 1:size(obj.cluster_centers_)[1]]
    labels = argmin(dist)
end

end
