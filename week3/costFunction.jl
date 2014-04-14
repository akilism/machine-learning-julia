# Function: costFunction(theta, X, y)
#
# Compute cost and gradient for logistic regression
# The cost of using theta as the parameter for logistic regression
# The gradient of the cost w.r.t. the parameters.
#
# Parameters:
#  theta - parameters for logistic regression
#  X - Matrix of examples
#  y - vector of classifications
#
# Returns:
#  J - cost
#  grad - gradient for each x.
#

function costFunction(theta, X, y)

  m = length(y)
  J = 0
  grad = zeros(size(theta))

  predictions = sigmoid(X*theta)

  results = y .* log(predictions) + (1 .- y) .* log(1 .- predictions)

  # Compute the cost.
  J = -(1/m) * sum(results)

  # Compute a gradient for each x.
  grad = (1/m) * sum(((predictions -y) .* X)', 2)

  return J, grad
end
