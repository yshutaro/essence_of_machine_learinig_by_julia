include("svm_hard.jl")

using .svm_hard
using Plots
using Random

Random.seed!(0)
#X0 = randn(20, 2)
X0 = [ 1.76405235  0.40015721;
 0.97873798  2.2408932 ;
 1.86755799 -0.97727788;
 0.95008842 -0.15135721;
-0.10321885  0.4105985 ;
 0.14404357  1.45427351;
 0.76103773  0.12167502;
 0.44386323  0.33367433;
 1.49407907 -0.20515826;
 0.3130677  -0.85409574;
-2.55298982  0.6536186 ;
 0.8644362  -0.74216502;
 2.26975462 -1.45436567;
 0.04575852 -0.18718385;
 1.53277921  1.46935877;
 0.15494743  0.37816252;
-0.88778575 -1.98079647;
-0.34791215  0.15634897;
 1.23029068  1.20237985;
-0.38732682 -0.30230275]
println("size X0:",size(X0))
#X1 = randn(20, 2) .+ [5 5]
X1 = [3.95144703 3.57998206;
3.29372981 6.9507754 ;
4.49034782 4.5619257 ;
3.74720464 5.77749036;
3.38610215 4.78725972;
4.10453344 5.3869025 ;
4.48919486 3.81936782;
4.97181777 5.42833187;
5.06651722 5.3024719 ;
4.36567791 4.63725883;
4.32753955 4.64044684;
4.18685372 3.2737174 ;
5.17742614 4.59821906;
3.36980165 5.46278226;
4.09270164 5.0519454 ;
5.72909056 5.12898291;
6.13940068 3.76517418;
5.40234164 4.31518991;
4.12920285 4.42115034;
4.68844747 5.05616534]
println("size X1:",size(X1))
y = vcat([1 for x in 1:20], [-1 for x in 1:20])
println(y)

println(X0)
println(X1)
X = vcat(X0, X1)
println("X: ", X)

model = svm_hard.SVC()
svm_hard.fit(model, X, y)

scatter(X0[:, 1], X0[:, 2], color="black", markershape=:+, label="")
scatter!(X1[:, 1], X1[:, 2], color="black", markershape=:star6, label="")

f(model, x) = (-model.w0_ - model.w_[1] * x) / model.w_[2]

x1 = -0.2
x2 = 6
#println("size(a_): ", size(model.a_))
println("[model.a_ .!= 0] :", [model.a_ .!= 0])
println("X: ", X)
println("size(X): ", size(X))
plot!([x1, x2], [f(model, x1), f(model, x2)], color="black", label="")
tf = model.a_ .!= 0
Xfalse = [false for x in 1:size(X)[1]]
println(X[hcat(tf, Xfalse)])
scatter!(X[hcat(tf, Xfalse)], X[hcat(Xfalse, tf)], color="red", markersize=10, markershape=:circle, markeralpha=0.1, label="")