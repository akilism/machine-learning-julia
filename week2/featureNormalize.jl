# Normalize the features in X.

function featureNormalize(X)

  X_norm = X
  mu = zeros(1, size(X, 2))
  sigma = zeros(1, size(X, 2))
  n = size(X, 2)

  # Get mean for each feature (col in matrix)
  mu = mean(X, 1)

  # Get standard deviation for each feature
  sigma = std(X, 1)

  # Compute the normalized value for each feature
  X_norm = (X .- mu) ./ sigma

  return X_norm, mu, sigma
end
