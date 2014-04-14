# Function oneVsAll(X, y, num_labels, lambda)
#
# Trains multiple logistic regression classifiers and returns all classifiers
# in a matrix all_theta.

function oneVsAll(X, y, num_labels, lambda)

  m = size(X, 1)
  n = size(X, 2)
  all_theta = zeros(num_labels, n + 1)
  X = [ones(m, 1) X]

  initial_theta = zeros(n + 1, 1)
  # Figure out julia way of doing this.
  # options = optimset('GradObj', 'on', 'MaxIter', 50);

  for c = [1:num_labels]
    # Figure out julia way of doing this.
    # [theta] = fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), initial_theta, options);
    all_theta[c, :] = theta[:, 1]
  end
end
