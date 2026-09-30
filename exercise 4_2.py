# EXERCISE 4.2
# Multiple Layer Perceptron with Three Inputs

from sklearn.neural_network import MLPClassifier

# Set up training data (3 inputs)
X = [[0., 0., 0.],
     [1., 1., 1.],
     [0., 0., 1.],
     [1., 0., 0.],
     [0., 1., 0.],
     [0., 1., 1.],
     [1., 0., 1.],
     [1., 1., 0.]]

# Target output (logical OR)
y = [0, 1, 1, 1, 1, 1, 1, 1]

# Create the neural network
clf = MLPClassifier(
    solver='lbfgs',
    alpha=1e-5,
    hidden_layer_sizes=(5, 2),
    random_state=1
)

# Train the model
clf.fit(X, y)

# Test the model with three inputs
print(clf.predict([
    [1., 0., 0.],
    [0., 0., 0.]
]))

# Display the shape of weights
print([coef.shape for coef in clf.coefs_])