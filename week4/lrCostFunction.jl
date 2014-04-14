# Function: lrCostFunction(theta, X, y, lambda)
#
# Compute cost and gradient for regularized logistic regression
# The cost of using theta as the parameter for regularized logistic regression
# The gradient of the cost w.r.t. the parameters.
#
# Parameters:
#  theta - parameters for logistic regression
#  X - Matrix of examples
#  y - vector of classifications
#  lambda - regularization parameter
#
# Returns:
#  J - cost
#  grad - gradient for each x.
#

function lrCostFunction(theta, X, y, lambda)

  m = length(y)
  J = 0
  grad = zeros(size(theta))

  # Compute predictions.
  predictions = sigmoid(X*theta)

  # Compute regularization of parameters for cost.
  regularization = (lambda / (2*m)) * sum(theta[2:length(theta)].^2)

  # Compute regularization of parameters for gradient.
  reg_grad = (lambda / m) * theta
  reg_grad[1] = 0

  results = y .* log(predictions) + (1 - y) .* log(1 - predictions)

  # Compute the cost.
  J = (-(1/m) * sum(results)) + regularization

  # Compute a gradient for each x.
  grad = ((1/m) * sum(((predictions -y) .* X)', 2)) + reg_grad

  return J, grad
end
