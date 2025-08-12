from data import get_training_data
from perceptron import Perceptron
import numpy as np


data = get_training_data()
X = data['dataset']
Y = data['targets']


p = Perceptron(input_size=4, learning_rate=0.1)
p.train(X, Y, epochs=10000)

new_customers = np.array ([
    [ 1, 0, 1, 0], 
    [ 1, 0, 0, 1],  
    [ 1, 0, 0, 0],
    [ 0, 0, 1, 0]
])

new_predictions = p.predict(new_customers)

print(new_predictions)

print(p.predict(X))




