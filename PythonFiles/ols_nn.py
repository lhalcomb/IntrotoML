#!/usr/bin/env python3
# ols_nn.py                                                   SSimmons Oct. 2018
"""
Uses the nn library along with auto-differentiation to define and train, using
batch gradient descent, a neural net that finds the ordinary least-squares
regression plane. The data are mean centered and normalized.  The plane obtained
via gradient descent is compared with the one obtained by solving the normal
equations.
"""
import csv
import torch
import torch.nn as nn

# Read in the data.
with open('./assignfiles/temp_co2_data.csv') as csvfile:
  reader = csv.reader(csvfile, delimiter=',')
  next(csvfile)  # skip the first line of csvfile
  xss, yss = [], []
  for row in reader:
    xss.append([float(row[2]), float(row[3])])
    yss.append([float(row[1])])

# The tensors xss and yss now containing the features (i.e., inputs) and targets
# (outputs) of the data.
xss, yss = torch.tensor(xss), torch.tensor(yss)

# For validation, compute the least-squares regression plane using linear alg.
A = torch.cat((torch.ones(len(xss),1), xss), 1)
lin_alg_sol = torch.linalg.lstsq(A, yss, driver='gels').solution[:,0]

# Compute the column-wise means and standard deviations.
# Comment out the next 2 lines, for example, to see what happens if you do not
# mean center; or comment out all 4 lines; or just the last 2.
xss_means = xss.mean(0)  # xss_means.size() returns torch.size([2])
yss_means = yss.mean()   # yss_means.size() returns torch.size([])
xss_stds  = xss.std(0)   # similarly here
yss_stds  = yss.std()    # and here

# Mean center the inputs and output (if xss_means and yss_means are defined).
if 'xss_means' in locals() and 'yss_means' in locals():
  xss, yss = xss - xss_means, yss - yss_means

# Normalize the inputs and output (if xss_stds and yss_stds are defined).
if 'xss_stds' in locals() and 'yss_stds' in locals():
  xss, yss = xss/xss_stds, yss/yss_stds

# Build the model
class LinearRegressionModel(nn.Module):

  def __init__(self):
    super(LinearRegressionModel, self).__init__()
    self.layer = nn.Linear(2,1)

  def forward(self, xss):
    return self.layer(xss) 

# Create an instance of the above class.
model = LinearRegressionModel()
print("The model is:\n", model)

# Set the criterion to be mean-squared error
criterion = nn.MSELoss()
"""
README!! 

I attempted to make the code a bit more readable in the for loop. 
I changed some variable names, swapped code to variables where needed, and 
added comments to explain what each section of code is doing. On top of 
getting the batchsize implememntation to work. 

- Layden H.
"""
learning_rate = 0.5
epochs = 30
batchsize = 32

num_example = len(xss) #32 inputs

#beginning implementation of momentum
momentum = 0

for epoch in range(epochs):  # train the model

  accum_loss = 0
  indicies = torch.randperm(num_example) #creates a random permutation of the example list
  

  """
  i.e.
  tensor([25, 27,  8,  0,  3, 17, 20, 26,  7,  9,  5,  1, 18, 10, 30, 21, 16, 12,
        28, 31, 24,  4, 13,  6,  2, 23, 15, 22, 14, 19, 29, 11])
  """
 

  for i in range(0, num_example, batchsize): #go from 0 to num_example per a step of batchsize

    #slice the data to get the current batch
    indicies_batch = indicies[i:i+batchsize] #random perm of size batchsize. eg. tensor([31,  8, 14,  6])

   
    xss_batch = xss[indicies_batch]
    """
    The data in the correspondant x columns output below: 
    xss_batch = tensor([[ 0.8815,  0.0238],
        [ 0.4513,  0.6580],
        [ 1.4111, -1.4398],
        [-0.0606, -0.8056]])
    """

    yss_batch = yss[indicies_batch]
    """The data in the correspondant y columns output below: 
       yss_batch = tensor([[ 1.1680],
        [-0.1415],
        [ 1.3317],
        [ 0.1859]])
    """

    # yss_pred refers to the outputs predicted by the model
    yss_pred = model(xss_batch) 

    loss = criterion(yss_pred, yss_batch) # compute the loss
    accum_loss += loss.item() # accumulate the loss

    #print("epoch: {0}, current loss: {1}".format(epoch+1, loss.item()))

    model.zero_grad() # set the gradient to the zero vector
    loss.backward() # compute the gradient of the loss function w/r to the weights

    #adjust the weights
    for param in model.parameters():
      param.data.sub_(param.grad.data * learning_rate) #w_i^{k+1} = w_i^{k} - alpha * grad(w_i^{k})
      #print(param.grad.data * learning_rate)
  
  print("epoch: {0}, current loss: {1}".format(epoch+1, accum_loss/(num_example//batchsize))) 

# extract the weights and bias into a list
params = list(model.parameters())
#print(params)

# Un-mean-center and un-normalize the weights (for final validation against
# the OLS regression plane found above using linear algebra).
if not ('xss_means' in locals() and 'yss_means' in locals()):
  xss_means = torch.Tensor([0.,0.]); yss_means = 0.0
  print('no centering, ', end='')
else:
  print('centered, ', end='')
if not ('xss_stds' in locals() and 'yss_stds' in locals()):
  xss_stds = torch.Tensor([1.,1.]); yss_stds = 1.0;
  print('no normalization, ', end='')
else:
  print('normalized, ', end='')
w = torch.zeros(3)
w[1:] = params[0] * yss_stds / xss_stds
w[0] = params[1].data.item() * yss_stds + yss_means - w[1:] @ xss_means

print("The least-squares regression plane:")
# Print out the equation of the plane found by the neural net.
print("  found by the neural net is: "+"y = {0:.3f} + {1:.3f}*x1 + {2:.3f}*x2"\
    .format(w[0],w[1],w[2]))

# Print out the eq. of the plane found using closed-form linear alg. solution.
print("  using linear algebra:       y = "+"{0:.3f} + {1:.3f}*x1 + {2:.3f}*x2"\
    .format(lin_alg_sol[0], lin_alg_sol[1], lin_alg_sol[2]))
print(f"learning rate: {learning_rate}")
print(f"batch size: {batchsize}")
