# Function sigmoid(z)
#
# Compute sigmoid of z
#
# Parameters:
#  z -  (real number, vector, matrix)
#
# Returns:
#  sigmoid value of every z.
#
# Test data:
# sigmoid([0; 1; 1; 0])
# sigmoid([0, 1, 1, 0])
# sigmoid([0, 0; 1, 1; 1, 19999999999999999; 1 0])
#

function sigmoid(z)

  return 1.0 ./ (1.0 .+ exp(-z))

end
