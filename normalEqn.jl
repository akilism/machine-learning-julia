# Computes closed form solution to linear regression.

function normalEqn(X, y)
  inv(X' * X) * X' * y
end
