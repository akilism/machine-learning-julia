

function predict(Theta1, Theta2, X)

  m = size(X, 1)
  num_labels = size(Theta2, 1)
  p = zeros(m, 1)

  # Set the input layer
  a1 = [ones(m, 1) X]

  # Compute the predictions
  z2 = a1 * Theta1';

  # Setup the first hidden layer. get size, compute h(x), add in col of 1s
  m = size(z2, 1)
  a2 = sigmoid(z2)
  a2 = [ones(m, 1) a2]

  # Compute predictions
  z3 = a2 * Theta2'

  a3 = sigmoid(z3)

  for i = [1:size(a3, 1)]
    # Get prediction per example (max value in each row)
    [val, idx] = findmax(a3[i])
    p[i] = idx
    h[i] = val
  end
  return p, h
end
