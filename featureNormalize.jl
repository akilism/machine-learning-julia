# Normalize the features in X.
function featureNormalize(X)

  X_norm = X
  mu = zeros(1, size(X, 2))
  sigma = zeros(1, size(X, 2))
  n = size(X, 2)

  for feature = [1:n]
    # Get mean for each feature (col in matrix)
    mu[feature] = mean(X[:, feature])

    # Get standard deviation for each feature
    sigma[feature] = std(X[:, feature])
  end

  # Compute the normalized value for each feature
  X_norm = (X .- mu) ./ sigma

  return X_norm, mu, sigma
end