include("logisticreg.jl")
using .logisticreg
using CSV

n_test = 100

dataframe = CSV.read("wdbc.data", header=false)
row,col=size(dataframe)

X = Float64[dataframe[r,c] for r in 1:row, c in 3:col]
y = Float64[ifelse(dataframe[r,2] == "B", 0, 1) for r in 1:row]

y_train = y[1:end-n_test]
X_train = X[1:row-n_test, :]
y_test = y[end-n_test+1:end]
X_test = X[end-n_test+1:end, :]

model = logisticreg.LogisticRegression(0.01)
logisticreg.fit(model, X_train, y_train)

y_predict = logisticreg.predict(model, X_test)
n_hits = sum(y_test .== y_predict)
println("Accuracy: $n_hits/$n_test = $(n_hits / n_test)")
