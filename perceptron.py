import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.weights = np.zeros(input_size)
        self.bias = 0.0

    def sigmoid(self, z):
        return  1 / (1 + np.exp(-z))

    def predict_probability(self, X):
        output = np.dot(X, self.weights) + self.bias
        return self.sigmoid(output)
    
    
    def predict (self, X):
        probability = self.predict_probability(X)
        return np.round(probability).astype(int).tolist()
    
    def train (self, X, Y, epochs):
        for i in range(epochs):
            z = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(z)
            error = predictions - Y
            loss = np.mean(error ** 2)
            gradient = error * predictions * (1 - predictions)
            self.weights -= self.learning_rate * np.dot(X.T, gradient)
            self.bias -= np.sum(gradient) * self.learning_rate
        

        
    
