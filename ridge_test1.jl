using .ridge
using Plots

x = [1, 2, 4, 6, 7]
y = [1, 3, 3, 5, 4]

model = ridge.RidgeRegression(1.)
ridge.fit(model, x, y)
b, a = model.w_

scatter(x, y, color="black")
xmax = maximum(x)
plot!([0, xmax], [b, b + a * xmax], color="black")

