module ridge
using LinearAlgebra

mutable struct RidgeRegression
  λ_
  w_
  function RidgeRegression(lambda=1.)
    new(lambda, Nothing)
  end
end

function fit(s::RidgeRegression, X, t)
  Xtil = hcat(ones(size(X)[1]), X)
  c =  Matrix{Float64}(I, size(Xtil)[2], size(Xtil)[2]) # 単位行列
  A = Xtil' * Xtil + s.λ_ * c
  b = Xtil' * t
  s.w_ = A \ b
end

function predict(s::RidgeRegression, X)
  Xtil = hcat(ones(size(X)[1]), X)
  return Xtil * s.w_
end

end
