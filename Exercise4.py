# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 23:14:12 2022

@author: Jungyu Lee, 301236221
Exercise 4
"""
import numpy as np
import neurolab as nl
import matplotlib.pyplot as plt


#1, 2. Create input data
input_Jungyu = np.random.uniform(-0.6, 0.6, (100,2))

#3. Create output data
output_Jungyu = input_Jungyu.sum(axis=1).reshape(100, 1)

#4. Set the seed = 1
np.random.seed(1)

#5. Create a neural network 
dim_1 = [-0.6, 0.6]
dim_2 = [-0.6, 0.6]
nn = nl.net.newff([dim_1, dim_2], [5, 3, 1])

# 4-3 Set the training algorithm to Gradient descent backpropogation
nn.trainf = nl.train.train_gd

#6, 7 Train the network
error_progress = nn.train(input_Jungyu, output_Jungyu, epochs=1000, show=100, goal=0.00001)

# 4-5 Plot the error 
plt.figure() 
plt.plot(error_progress)
plt.xlabel('Number of epochs')
plt.ylabel('Error')
plt.title('Training error progress')

data_test = [[0.1, 0.2]]
sim = nn.sim(data_test)
print('\nSimulation Result:', sim)
