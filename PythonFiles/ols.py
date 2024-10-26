#!/usr/bin/env python3
# ols.py                                                     SSimmons March 2018
"""
Uses a neural net to find the ordinary least-squares regression model. Trains
with batch gradient descent, and computes r^2 to gauge predictive quality.
"""

import torch
import pandas as pd
import torch.nn as nn

# Read the named columns from the csv file into a dataframe.
names = ['SalePrice','1st_Flr_SF','2nd_Flr_SF','Lot_Area','Overall_Qual',
    'Overall_Cond','Year_Built','Year_Remod/Add','BsmtFin_SF_1','Total_Bsmt_SF',
    'Gr_Liv_Area','TotRms_AbvGrd','Bsmt_Unf_SF','Full_Bath']
df = pd.read_csv('assignfiles/AmesHousing.csv', names = names)
data = df.values # read data into a numpy array (as a list of lists)
data = data[1:] # remove the first list which consists of the labels
data = data.astype(float) # coerce the entries in the numpy array to floats
data = torch.FloatTensor(data) # convert data to a Torch tensor

data.sub_(data.mean(0)) # mean-center
data.div_(data.std(0))  # normalize

# 2263 x 14 size tensor 
xss = data[:,1:] #read in all of the inputs after the first column
yss = data[:,:1] #read in the outputs before the first columns

# define a model class
class LinearModel(nn.Module):

  def __init__(self):
    super(LinearModel, self).__init__()
    self.layer1 = nn.Linear(13, 1)

  def forward(self, xss):
    return self.layer1(xss)

# create and print an instance of the model class
model = LinearModel()
print(model)

criterion = nn.MSELoss()

z_parameters = []
for param in model.parameters():
  z_parameters.append(param.data.clone())
for param in z_parameters:
  param.zero_()

num_examples = len(data) # 2264
batch_size = 30
learning_rate = .0001
epochs = 1000
momentum = 0.999


# train the model
for epoch in range(epochs):
  indices = torch.randperm(num_examples)
  for i in range(0, num_examples, batch_size):
    # randomly pick batchsize examples from data
    

    yss_mb = yss[indices[i: i + batch_size]]  # the targets for the mb (minibatch)
    yhatss_mb = model(xss[indices[i: i + batch_size]])  # model outputs for the mb

    loss = criterion(yhatss_mb, yss_mb)
    model.zero_grad()
    loss.backward() # back-propagate

    # update weights
    for i, (z_param, param) in enumerate(zip(z_parameters, model.parameters())):
      z_parameters[i] = momentum * z_param + param.grad.data
      param.data.sub_(z_parameters[i] * learning_rate)

  with torch.no_grad():
    total_loss = criterion(model(xss), yss).item()
  print('epoch: {0}, loss: {1:11.8f}'.format(epoch+1, total_loss))

print("total number of examples:", num_examples, end='; ')
print("batch size:", batch_size)
print("learning rate:", learning_rate)
print("momentum: ", momentum)

print_str = "epoch: {0}, loss: {1}".format(epoch+1, total_loss)

if epochs < 20 or epoch < 7 or epoch > epochs - 17:
    print(print_str)
elif epoch == 7:
    print("...")
else:
    print(print_str, end='\b' * len(print_str), flush=True)

# Compute 1-SSE/SST which is the proportion of the variance in the data
# explained by the regression hyperplane.
SS_E = 0.0;  SS_T = 0.0
mean = yss.mean()
for xs, ys in zip(xss, yss):
  SS_E = SS_E + (ys - model(xs))**2
  SS_T = SS_T + (ys - mean)**2
print(f"1-SSE/SST = {1.0-(SS_E/SS_T).item():1.4f}")
