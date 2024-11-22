import torch
#import pandas as pd
import torch.nn as nn
import du.lib  as dulib
from skimage import io
from numpy import set_printoptions
from matplotlib import pyplot as plt

class ConvolutionalModel(nn.Module):

  def __init__(self):
    super(ConvolutionalModel, self).__init__()
    self.meta_layer1 = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding = 2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)
    )
    self.meta_layer2 = nn.Sequential(
      nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride = 1, padding = 2),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0 )
    )
    self.fc_layer1 = nn.Linear(800,200)
    self.fc_layer2 = nn.Linear(200,10)

  def forward(self, xss):
    xss = torch.unsqueeze(xss, dim=1)
    xss = self.meta_layer1(xss)
    xss = self.meta_layer2(xss)
    xss = torch.reshape(xss, (-1, 800))
    xss = self.fc_layer1(xss)
    xss = self.fc_layer2(xss)
    return torch.log_softmax(xss, dim=1)

# create an instance of the model class
model = ConvolutionalModel()

# set the criterion
criterion = nn.NLLLoss()


def subset_data(num_examples, partition):
  # partition the data into training and test sets

  train_size = int(num_examples * partition)
  indices = torch.randperm(num_examples)

  #with partition = 0.8
  train_data = indices[:train_size] # 80% 
  test_data =  indices[train_size:] # 20%

  print(f'training on: {len(train_data)}, testing on: {len(test_data)}')

  return train_data, test_data


# read in all of the digits
digits = io.imread('./assignfiles/digits.png')
xss = torch.Tensor(5000, 20, 20)
idx = 0
for i in range(0, 1000, 20):
  for j in range(0, 2000, 20):
    xss[idx] = torch.Tensor((digits[i:i+20,j:j+20]))
    idx = idx + 1


xss, xss_means = dulib.center(xss)
#xss, xss_stds = dulib.normalize(xss) 

# generate yss to hold the correct classification for each example
yss = torch.LongTensor(len(xss))
for i in range(len(yss)):
  yss[i] = i//500



#implementing momentum
z_parameters = []
for param in model.parameters():
  z_parameters.append(param.data.clone())
for param in z_parameters:
  param.zero_()

learning_rate = 0.0001
epochs = 1000
batchsize = 16

num_example = len(xss) #5000 inputs

#collect train and test data
partition = 0.80
train_data, test_data = subset_data(num_example, partition)

#set the correspondent test data
xss_train, yss_train = xss[train_data], yss[train_data]
xss_test, yss_test = xss[test_data], yss[test_data] 

#beginning implementation of momentum
momentum = 0.9


""" 
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
params = list(model.parameters()) """


model = dulib.train(
        model = model,
        crit = criterion,
        train_data = (xss_train, yss_train),
        valid_data = (xss_test, yss_test),
        learn_params = {'lr': learning_rate, 'mo':  momentum},
        bs =  batchsize, 
        epochs = epochs, 
        
)

""" count = 0
for i in range(len(xss_test)):
  if  torch.argmax(model(xss_test[i].unsqueeze(0))).item() == yss_test[i].item():
    count += 1 """

#correct on train set
pct_train = dulib.class_accuracy(model, (xss_train, yss_train), show_cm=False)
print(f"Percentage correct on training data: {100*pct_train:.2f}")

#correct on test set
pct_test = dulib.class_accuracy(model, (xss_test, yss_test), show_cm=True)
print(f"Percentage correct on testing data: {100*pct_test:.2f}")



