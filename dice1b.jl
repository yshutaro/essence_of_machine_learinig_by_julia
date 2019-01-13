using Random

function throw_dice(n, random_seed = 10)
  Random.seed!(random_seed)
  sum(rand(1:6, n))
end
