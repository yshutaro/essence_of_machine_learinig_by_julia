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

plt_l = []
plt_r = []

for i in 0:4
    xx = x[1:2 + i * 2]
    yy = x[1:2 + i * 2]
    plt_s1= scatter(xx, yy, color="black", xlim=(xmin, xmax), ylim=(ymin, ymax))

    model = linearreg.LinearRegression()
    linearreg.fit(model, xx, yy)
    xs = [xmin, xmax]
    ys = [model.w_[1] + model.w_[2] * xmin,
    model.w_[1] + model.w_[2] * xmax]

    push!(plt_l,plot!(plt_s1, xs, ys, color="black"))

    model = ridge.RidgeRegression(10.)
    ridge.fit(model, xx,yy)
    xs = [xmin, xmax]
    ys = [model.w_[1] + model.w_[2] * xmin,
    model.w_[1] + model.w_[2] * xmax]

    plt_s2 = scatter(xx, yy, color="black", xlim=(xmin, xmax), ylim=(ymin, ymax))
    push!(plt_r, plot!(plt_s2, xs, ys, color="black"))

end

plot(plt_l[1], plt_l[2], plt_l[3], plt_l[4], plt_l[5],
     plt_r[1], plt_r[2], plt_r[3], plt_r[4], plt_r[5],
     layout=(2,5))
