include("./kmeans.jl")
using .kmeans
using Random
using Plots

Random.seed!(0)
points1 = randn(50, 2)
points2 = randn(50, 2) .+ [5 0]
points3 = randn(50, 2) .+ [5 5]

points_noshuffle = vcat(points1, points2, points3)
points = points_noshuffle[shuffle(1:end), :]

model = kmeans.KMeans(3)
kmeans.fit(model, points)

p1 = points[model.labels_ .== 1, :]
p2 = points[model.labels_ .== 2, :]
p3 = points[model.labels_ .== 3, :]

scatter(p1[:, 1], p1[:, 2], color=:black, markershape=:+)
scatter!(p2[:, 1], p2[:, 2], color=:black, markershape=:star6)
scatter!(p3[:, 1], p3[:, 2], color=:black, markershape=:circle)
