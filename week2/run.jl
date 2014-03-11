include("computeCost.jl");
include("gradientDescent.jl");
include("featureNormalize.jl");
include("normalEqn.jl");

function run()

  print_with_color(:green, "Loading data ...\n")
  fileStream = open("ex1data2.txt", "r")
  data = readcsv(fileStream)

  print_with_color(:green, "Running gradient descent...\n")
  runGradientDescent(data)

  print_with_color(:red, "Compute Normal Equations (Y/n): ");
  runStd = readline()

  if lowercase(runStd) != "n"
    print_with_color(:green, "\nComputing normal equations...\n")
    runNormalEquations(data)
  end

  print_with_color(:green, "Bye!")
end


function runGradientDescent(data)
  ##############################################
  # n = number of features in dataset.
  # m = number of training examples in dataset.
  # y = vector of output values.
  # X = n x m matrix of input values.
  ##############################################

  # Number of features in data is columns - 1.
  n = size(data)[:2] - 1

  # X is columns 1 -> columns-1
  X = data[:, 1:n]

  # y is always last column in data.
  y = data[:, [n + 1]]
  m = length(y)

  # Normalize features also return mu and sigma to do further predictions.
  X, mu, sigma = featureNormalize(X);

  # Add intercept (column of 1's).
  X = [ones(m, 1) X]

  # Set an alpha.
  alpha = 0.1
  iterations = 1500

  # Initialize theta (n + 1 vector).
  theta = zeros(n + 1, 1)

  theta, jHistory = gradientDescent(X, y, theta, alpha, iterations)

  # Display gradient descent's result
  print_with_color(:blue, "Theta computed from gradient descent: \n")
  println(theta)

  # Estimate the price of a 1650 sq-ft, 3 br house
  price = [1, (1650 - mu[1])/sigma[1], (3-mu[2])/sigma[2]]' * theta
  print_with_color(:blue, "Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):\n")
  println(round(price, 2))

  print_with_color(:red, "Square Feet: ");
  sqft = int(readline())

  print_with_color(:red, "# Bedrooms: ");
  beds = int(readline())

  price = [1, (sqft - mu[1])/sigma[1], (beds - mu[2])/sigma[2]]' * theta
  print_with_color(:blue, "Predicted price of a $(sqft) sq-ft, $(beds) br house (using gradient descent):\n")
  println(round(price, 2))
  print("\n")
end


function runNormalEquations(data)
  # Number of features in data is columns - 1.
  n = size(data)[:2] - 1

  # X is columns 1 -> columns-1
  X = data[:, 1:n]

  # y is always last column in data.
  y = data[:, [n + 1]]
  m = length(y)

  # Add intercept (column of 1's).
  X = [ones(m, 1) X]

  # Calculate the parameters from the normal equations
  theta = normalEqn(X, y)

  # Display normal equation's result
  print_with_color(:blue, "Theta computed from the normal equations: \n")
  println(theta)


  # Estimate the price of a 1650 sq-ft, 3 br house
  price = [1, 1650, 3]' * theta

  print_with_color(:blue, "Predicted price of a 1650 sq-ft, 3 br house (using normal equations):\n")
  println(round(price, 2))

  print_with_color(:red, "Square Feet: ");
  sqft = int(readline())

  print_with_color(:red, "# Bedrooms: ");
  beds = int(readline())

  price = [1, sqft, beds]' * theta
  print_with_color(:blue, "Predicted price of a $(sqft) sq-ft, $(beds) br house (using normal equations):\n")
  println(round(price, 2))
  print("\n\n")
end
