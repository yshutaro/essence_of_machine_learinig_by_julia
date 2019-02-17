using .linearreg
using CSV
using Random
using Statistics
using Printf

dataframe = CSV.read("winequality-red.csv", header=true, delim=';')
row,col=size(dataframe)

Xy = Float64[dataframe[r,c] for r in 1:row, c in 1:col]

Random.seed!(0)
shuffle(Xy)

train_X = Xy[1:row-1000, 1:col-1]
train_y = Xy[1:row-1000, col-1]
test_X = Xy[row-999:row, 1:col-1]
test_y = Xy[row-999:row, col-1]

model = linearreg.LinearRegression()
linearreg.fit(model, train_X, train_y)

y = linearreg.predict(model, test_X)

println("最初の5つの正解と予測値:")
for i in 1:5
  println("$(@sprintf("%1.0f", test_y[i])) $(@sprintf("%5.3f",y[i]))")
end
println()
println("RMSE:", mean(sqrt.((test_y .- y ).^2)))

