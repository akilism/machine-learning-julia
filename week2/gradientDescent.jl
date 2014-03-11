# Preform gradient descent to compute theta.
# theta =  number of iterations steps of gradient descent with the learning rate alpha.

function gradientDescent(X, y, theta, alpha, iterations)

  m = length(y)
  jHistory = zeros(iterations, 1)

  for step = [1:iterations]
    # delta is the sum of (h(X(i)) - y(i)) * X(i) / m
    # sum((X * theta - y) .* X)
    # theta = theta - alpha * delta

    val1 = (X * theta) - y

    # sum of (h(X(i)) - y(i)) * X(i)
    val = sum(val1 .* X, 1)

    delta = (1 / m) * val
    theta = theta - (alpha * delta)'

    # Save the cost J in every iteration
    jHistory[step] = computeCost(X, y, theta)
    # println(jHistory[step])
  end

  return theta, jHistory
end
