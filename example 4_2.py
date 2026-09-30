# Example 4.2 Multiple Layer Perceptron

from sklearn.neural_network import MLPClassifier

# Set up training data
X = [[0., 0.],
     [1., 1.],
     [0., 1.],
     [1., 0.]]

y = [0, 1, 1, 1]

# Create the neural network
clf = MLPClassifier(
    solver='lbfgs',
    alpha=1e-5,
    hidden_layer_sizes=(5, 2),
    random_state=1
)

# Train the model
clf.fit(X, y)

# Test the model
print(clf.predict([[2., 2.], [-1., -2.]]))

# Display the shape of weights
print([coef.shape for coef in clf.coefs_])