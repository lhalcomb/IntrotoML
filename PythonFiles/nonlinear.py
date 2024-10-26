#!/usr/bin/env python3
# nonlinear.py                                                    SSimmons March 2018 - Halcomb Fall Semester 2024
"""
Uses a neural net to find the ordinary least-squares regression model. Trains
with batch gradient descent, and computes r^2 to gauge predictive quality.

This now trains on a nonlinear model using relu and has implemented a split
in test and train data for cross-validation. - Layden H. 
"""

import torch
import pandas as pd
import torch.nn as nn
import du.lib  as dulib

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
class NonLinearModel(nn.Module):

  def __init__(self):
    super(NonLinearModel, self).__init__()
    self.layer1 = nn.Linear(13, 10)
    self.layer2 = nn.Linear(10, 1)


  def forward(self, xss):
    
    xss = self.layer1(xss)
    xss = torch.relu(xss)
    #xss = torch.sigmoid(xss)

    return self.layer2(xss)
  
def subset_data(num_examples, partition):
  # partition the data into training and test sets

  train_size = int(num_examples * partition)
  indices = torch.randperm(num_examples)

  #with partition = 0.8
  train_data = indices[:train_size] # 80% 
  test_data =  indices[train_size:] # 20%

  print(f'training on: {len(train_data)}, testing on: {len(test_data)}')

  return train_data, test_data



# create and print an instance of the model class
model = NonLinearModel()

criterion = nn.MSELoss()

z_parameters = []
for param in model.parameters():
  z_parameters.append(param.data.clone())
for param in z_parameters:
  param.zero_()

num_examples = len(data) # 2264

#collect train and test data
partition = 0.883
train_data, test_data = subset_data(num_examples, partition)

#set the correspondent test data
xss_train, yss_train = xss[train_data], yss[train_data]
xss_test, yss_test = xss[test_data], yss[test_data] 


batch_size = 32
learning_rate = .0001
epochs = 1000
momentum = 0.99



# train the model
for epoch in range(epochs):
  #indices = torch.randperm(num_examples)

  #takes random permutation of the train partition on the x values in the tensor
  train_perm = torch.randperm(len(xss_train))  
  

  for i in range(0, len(xss_train), batch_size):
    # randomly pick batchsize examples from data
    
    mb_indices = train_perm[i: i + batch_size] #mini batch indices targets on the train
    yss_mb = yss_train[mb_indices]  # the targets for the mb (minibatch)
    yhatss_mb = model(xss_train[mb_indices])  # model outputs for the mb

    loss = criterion(yhatss_mb, yss_mb) #compute the loss 
    model.zero_grad() #zero out the gradient
    loss.backward() # back-propagate and collect the gradient

    # update weights based on gradient and us momentum, use learning rate
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

#prints the proportion of the variance in the data explained by the regression hyperplane
print(f'explained variation on {len(train_data)}, {partition * 100}% for the train data: {dulib.explained_var(model, (xss_train, yss_train))}')
print(f'explained variation on {len(test_data)}, {100 - (partition * 100)}% for the test data: {dulib.explained_var(model, (xss_test, yss_test))}')
""" 
model = dulib.train(
  model, 
  crit = nn.MSELoss(),
  train_data=  (xss_train, yss_train),
  valid_data = (xss_test,  yss_test),
  #valid_metric = True,
  learn_params = {'lr': 0.006,  'mo': 0.96},
  epochs = 200,
  bs = 25,
  graph = 1
) """