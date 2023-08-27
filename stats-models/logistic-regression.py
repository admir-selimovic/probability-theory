import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Define the sigmoid function for logistic transformation
def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

# Define the logistic cost function
def cost_function(predictions, labels):
    cost = (-labels * np.log(predictions) - (1 - labels) * np.log(1 - predictions)).mean()
    return cost

# Update the weights using gradient descent
def update_weights(features, labels, weights, lr):
    predictions = predict(features, weights)
    gradient = np.dot(features.T, predictions - labels) / features.size
    weights -= lr * gradient
    return weights

# Train the logistic regression model
def train(features, labels, lr, iters):
    cost_history = []
    weights = np.zeros(features.shape[1])  # Initialize weights to zeros

    for i in range(iters):
        weights = update_weights(features, labels, weights, lr)
        predictions = predict(features, weights)
        cost = cost_function(predictions, labels)
        cost_history.append(cost)
        if i % 10000 == 0:
            print(f'iteration: {i}; cost: {cost}')

    return weights, cost_history, predictions

# Load the Iris dataset and extract the first two features
iris = load_iris()
features = iris.data[:, :2]
labels = (iris.target != 0) * 1  # Convert target labels to binary

# Set learning rate and number of iterations, then train the model
lr = 0.001
iters = 100000
out = train(features, labels, lr, iters)

# Extract the cost history for plotting
cost_h = out[1]

# Plot the cost history
plt.style.use('seaborn-white')
plt.figure(figsize=(7, 7))
plt.plot(cost_h, color='skyblue', linewidth=1.0)
plt.ylabel('cost', fontsize=10)
plt.show()

# Convert predicted probabilities to binary labels using threshold 0.5
predicted_labels = (out[2] >= 0.5).astype(int)

# Calculate and print the accuracy of the model
accuracy_value = 1.0 - np.mean(np.abs(predicted_labels - labels))
print("Accuracy:", accuracy_value)
