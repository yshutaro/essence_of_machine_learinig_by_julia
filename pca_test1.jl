include("./pca.jl")
using .pca
using CSV
using Plots

dataframe = CSV.read("winequality-red.csv", header=true, delim=';')
row,col=size(dataframe)

Xy = Float64[dataframe[r,c] for r in 1:row, c in 1:col]
X = Xy[:, 1:end-1]

model = pca.PCA(2)
pca.fit(model, X)

Y = pca.transform(model, X)
scatter(Y[:,1], Y[:,2], color=:black)
