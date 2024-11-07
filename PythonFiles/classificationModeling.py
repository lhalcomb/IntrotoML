import torch
#import pandas as pd
import torch.nn as nn
import du.lib  as dulib
from skimage import io
from numpy import set_printoptions



digits = io.imread('./assignfiles/digits.png')  # digits is now a numpy ndarray

""" xs = digits[:20,:20] #  xs is now a numpy ndarray of shape 20x20

set_printoptions(linewidth=100)
print(xs.flatten()) """

class LinearModel(nn.Module):  # create the model class, subclassing nn.Module

   def __init__(self):
     super(LinearModel, self).__init__()
     self.layer1 = nn.Linear(400, 1)

   def forward(self, x):
     x = self.layer1(x)
     return x

model = LinearModel() # create an instance of the model class

criterion = nn.MSELoss() # create an instance of the PyTorch class nn.MSELoss

# read in all of the digits
digits = io.imread('./assignfiles/digits.png')
xss = torch.Tensor(5000,400)
idx = 0
for i in range(0, 1000, 20):
  for j in range(0, 2000, 20):
    xss[idx] = torch.Tensor((digits[i:i+20,j:j+20]).flatten())
    idx = idx + 1

# extract just the zeros and eights from xss
tempxss = torch.Tensor(1000,400)
tempxss[:500] = xss[:500]
tempxss[500:] = xss[4000:4500]

# overwrite the original xss with just zeros and eights
xss = tempxss

xss, xss_means = dulib.center(xss)
xss, xss_stds = dulib.normalize(xss)

# generate yss to hold the correct classification for each example
yss = torch.Tensor(len(xss),1)
for i in range(len(yss)):
  yss[i] = i//500 + 8

#implementing momentum
z_parameters = []
for param in model.parameters():
  z_parameters.append(param.data.clone())
for param in z_parameters:
  param.zero_()

learning_rate = 0.0001
epochs = 1500
batchsize = 32

num_example = len(xss) #1000 inputs

#beginning implementation of momentum
momentum = 0.9


for epoch in range(epochs):  # train the model

  accum_loss = 0
  indicies = torch.randperm(num_example) #creates a random permutation of the example list
 

  for i in range(0, num_example, batchsize): #go from 0 to num_example per a step of batchsize

    #slice the data to get the current batch
    indicies_batch = indicies[i:i+batchsize] #random perm of size batchsize. eg. tensor([31,  8, 14,  6])

   
    xss_batch = xss[indicies_batch]


    yss_batch = yss[indicies_batch]


    # yss_pred refers to the outputs predicted by the model
    yss_pred = model(xss_batch) 

    loss = criterion(yss_pred, yss_batch) # compute the loss
    accum_loss += loss.item() # accumulate the loss

    #print("epoch: {0}, current loss: {1}".format(epoch+1, loss.item()))

    model.zero_grad() # set the gradient to the zero vector
    loss.backward() # compute the gradient of the loss function w/r to the weights

    #adjust the weights
    for i, (z_param, param) in enumerate(zip(z_parameters, model.parameters())):
      z_parameters[i] = momentum * z_param + param.grad.data
      param.data.sub_(z_parameters[i] * learning_rate)
  
  print("epoch: {0}, current loss: {1}".format(epoch+1, accum_loss/(num_example//batchsize))) 

# extract the weights and bias into a list
params = list(model.parameters())


def pct_correct(yhatss, yss):
  zero = torch.min(yss).item()
  eight = torch.max(yss).item()
  th = 1e-3 # threshold
  cutoff = (zero+eight)/2
  count = 0

  for yhats, ys in zip(yhatss, yss):
    yhat = yhats.item()
    y = ys.item()
    if (yhat>cutoff and abs(y-eight)<th) or (yhat<cutoff and abs(y-zero)<th):
      count += 1

  return 100*count/len(yss)


def printDeClassifiedDigit(yhatss, yss, xss_original):
  zero  = torch.min(yss).item() #8
  eight = torch.max(yss).item() #9

  th =  1e-3 # threshold  
  cutoff = (zero + eight)/2

  classified_indicies = []

  for idx, (yhats, ys) in enumerate(zip(yhatss, yss)):
    yhat = yhats.item()
    y = ys.item()
    if (yhat > cutoff and abs(y - eight) < th ) or (yhat < cutoff and abs(y - zero)<th):
      classified_indicies.append(idx)
    else: 
      set_printoptions(linewidth=100)
      print(f"\nMisclassified digit at index {idx}:")
      print(xss_original[idx].reshape(20, 20).numpy().astype(int) )

  
  



""" model = dulib.train(
        model = model,
        crit = criterion,
        train_data = (xss, yss),
        valid_metric = pct_correct,
        learn_params = {'lr': learning_rate, 'mo':  momentum},
        bs =  batchsize, 
        epochs = 1000

) """
print("Percentage correct:", pct_correct(model(xss), yss))

printDeClassifiedDigit(model(xss), yss, xss)