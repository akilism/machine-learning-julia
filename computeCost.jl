# Compute cost for linear regression with multiple variables.

function computeCost(X, y, theta)

  m = length(y)

  J = 0

  # Get predictions for the whole matrix of X values.
  predictions = X * theta

  # Calculate squared errors
  squaredErrors = (predictions-y) .^ 2

  # Compute cost
  J = (1 / (2 * m)) * sum(squaredErrors)

  return J

end
