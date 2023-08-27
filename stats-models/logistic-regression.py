# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Define the sigmoid function for logistic transformation
def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

# Predict probabilities using the logistic model
def predict(features, weights):
    z = np.dot(features, weights)
    return sigmoid(z)

# Compute the logistic cost function
def cost_function(features, labels, weights):
    predictions = predict(features, weights)
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
    # Initialize weights to zeros
    weights = np.zeros(features.shape[1])

    for i in range(iters):
        weights = update_weights(features, labels, weights, lr)
        # Calculate cost for auditing purposes
        cost = cost_function(features, labels, weights)
        cost_history.append(cost)
        # Log Progress (print cost every 10,000 iterations)
        if i % 10000 == 0:
            print ('iteration:', i, '; cost:', cost)

    predicted_probabilities = predict(features, weights)

    return weights, cost_history, predicted_probabilities

# Load the Iris dataset and extract the first two features
iris = load_iris()
features = iris.data[:, :2]

# Convert target labels to binary: 0 if not class 0, 1 if class 0
labels = (iris.target != 0) * 1

# Set learning rate and number of iterations, then train the model
lr = 0.001
iters = 100000
out = train(features, labels, lr, iters)

# Extract the cost history for plotting
cost_h = []
for i in range(len(out[1])):
    cost_h.append(out[1][i].tolist())

# Plot the cost history
plt.style.use('seaborn-white')
plt.figure(figsize=(7,7))
plt.plot(cost_h, color='skyblue', linewidth=1.0)
plt.ylabel('cost', fontsize=10)
plt.show()

# Extract predicted probabilities from the output
predicted_probabilities = []
for i in range(len(out[2])):
    predicted_probabilities.append(out[2][i].tolist())

# Convert predicted probabilities to binary labels using threshold 0.5
predicted_labels = []
for i in range(len(predicted_probabilities)):
    if predicted_probabilities[i] >= 0.5:
        temp_class = 1
    else:
        temp_class = 0
    predicted_labels.append(temp_class)

predicted_labels = np.array(predicted_labels)

# Define a function to compute accuracy
def accuracy(predicted_labels, actual_labels):
    diff = predicted_labels - actual_labels
    return 1.0 - (float(np.count_nonzero(diff)) / len(diff))

# Calculate and print the accuracy of the model
accuracy_value = accuracy(predicted_labels, labels)
print("Accuracy:", accuracy_value)
