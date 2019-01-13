function softplus2(x)
  max(0, x) + log(1 + exp(-abs(x)))
end
