# Function: predict(theta, X)
#
# Predict if a the label is 0 or 1 using learned logistic regression parameters
# theta. If sigmoid(theta'*x) >= 0.5 prediction == 1  (0.5 threshold)
#
# Parameters:
#  theta - parameters for logistic regression
#  X - matrix of examples.
#
# Returns:
#  p - predictions for each x.
#

function predict(theta, X)

  m = size(X, 1)
  p = zeros(m, 1)
  thresholds = sigmoid(X*theta)

  # Check threshold for each x.
  for x = [1:m]
    if thresholds[x] >= 0.5
      p[x] = 1
    else
      p[x] = 0
    end
  end

  return p
end
