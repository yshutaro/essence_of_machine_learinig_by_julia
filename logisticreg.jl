module logisticRegression

const THRESHMIN = 1e -10

sigmoid(x) = 1 / (1 + ℯ^(-1))

struct LogisticRegression
    tol
    max_iter
    w_
    function LogisticRegression(tol, max_iter=3, random_seed=0)
        new(tol, max_iter, Nothing)
    end
end

function fit(s::LogisticRegression, X, y)
end

function predict(s::LogisticRegression, X)
end

end