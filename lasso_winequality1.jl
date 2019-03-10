include("lasso.jl")
using .lasso
using CSV
using Random
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

for λ_ in [1., 0.1, 0.01]
  model = lasso.Lasso(λ_)
  lasso.fit(model, train_X, train_y)
  y = lasso.predict(model, test_X)
  println("--- λ = $(λ_) ---")
  println("coefficients:")
  println(model.w_)
  mse = mean((y - test_y).^2)
  println("MSE:$(@sprintf("%0.3f", mse))")
end
