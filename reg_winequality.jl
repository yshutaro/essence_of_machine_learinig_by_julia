using .linearreg
using CSV
using Random

dataframe = CSV.read("winequality-red.csv", header=true, delim=';')
row,col=size(dataframe)

Xy = Float64[dataframe[r,c] for r in 1:row, c in 1:col]

Random.seed!(0)
shuffle(Xy)
