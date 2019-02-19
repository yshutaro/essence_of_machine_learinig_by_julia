using .ridge
using .linearreg
using Plots

x = collect(0:11)
y = 1 .+ 2 .* x
y[3] = 20
y[5] = 0

xmin = 0
xmax = 12
ymin = -1
ymax = 25


xx = x[1:2 + 0 * 2]
yy = x[1:2 + 0 * 2]

plt1 = scatter(xx, yy, color="black", xlim=(xmin, xmax), ylim=(ymin, ymax))

model = linearreg.LinearRegression()
linearreg.fit(model, xx, yy)
xs = [xmin, xmax]
ys = [model.w_[1] + model.w_[2] * xmin,
model.w_[1] + model.w_[2] * xmax]
plt2 = plot!(plt1, xs, ys, color="black")

model = ridge.RidgeRegression(10.)
ridge.fit(model, xx,yy)
xs = [xmin, xmax]
ys = [model.w_[1] + model.w_[2] * xmin,
model.w_[1] + model.w_[2] * xmax]

plt3 = scatter(xx, yy, color="black", xlim=(xmin, xmax), ylim=(ymin, ymax))
plt4 = plot!(plt3, xs, ys, color="black")

plot(plt2, plt4, laout=(2,1))

