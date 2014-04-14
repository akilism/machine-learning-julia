include("costFunction.jl");
include("costFunctionReg.jl");
include("predict.jl");
include("sigmoid.jl");

function run()

  print_with_color(:green, "Loading data ...\n")
  fileStream = open("projects/machinelearning_coursera/class_work/julia/week3/ex2data1.txt", "r")
  data = readcsv(fileStream)
  X = data[:, [1, 2]]
  y = data[:, 3]

  print_with_color(:green, "Running Computing cost and gradient at initial theta...\n")
  computeCostAndGradient(X, y)

  print_with_color(:green, "Bye!")
end

########################################################
# Function: computeCostAndGradient(X, y)
#
# Computes cost and gradient for inital_theta (all 0's)
#
# Parameters:
#  X - matrix of examples
#  y - labels
#
########################################################
function computeCostAndGradient(X, y)

  m = size(X, 1)
  n = size(X, 2)

  X = [ones(m, 1) X]

  initial_theta = zeros(n + 1, 1)

  cost, grad = costFunction(initial_theta, X, y)

  print_with_color(:yellow, "Cost at initial theta (0's): $(cost)\n")
  print_with_color(:yellow, "Gradient at initial theta (0's): \n $(grad)\n")

end

run()
