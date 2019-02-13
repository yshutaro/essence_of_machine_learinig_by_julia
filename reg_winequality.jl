using .linearreg
using CSV

Xy = []
dataframe = CSV.read("winequality-red.csv", header=true, delim=';')
@show dataframe
row,col=size(dataframe)
for  c =1:col
    push!(Xy, c)
end
@show dataframe