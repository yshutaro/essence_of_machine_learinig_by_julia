function qeq2(a, b, c)
  alpha = (-b - sign(b) * sqrt(b^2 - 4 * a * c)) / (2 * a)
  beta = c / (a * alpha)
  (alpha, beta)
end
