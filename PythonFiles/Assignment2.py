import torch
import csv
import matplotlib.pyplot as plt

with open('assignfiles/assign2.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter = ',')
    next(csvfile) #skip the first line
    xs, ys = [], []
    for row in reader:
        xs.append(float(row[0]))
        ys.append(float(row[1]))
    
xs, ys = torch.tensor(xs), torch.tensor(ys)

ys = ys.unsqueeze(1) # we moved this demension from 1d to 2d to use the mm() operation, everytihing in r2
xTensor = torch.ones(60,2)

xTensor[:, 1] = xs

omegaWeights = xTensor.transpose(0,1).mm(xTensor).inverse().mm(xTensor.transpose(0,1)).mm(ys)


if __name__ == "__main__":
    
    """
    #debugging and testing

    print(xTensor)
    print(ys)
    print(xs)
    """
    print(omegaWeights)



   #Scatter Plot 
    plt.scatter(xs, ys, label= "Data Points", color="blue")

    """
    Plot the regression line
    omegaWeights = [bias, slope], so the regression line is: y = bias + slope * x
    """
    regression_line = omegaWeights[0].item() + omegaWeights[1].item() * xs
    plt.plot(xs, regression_line, label="Regression line", color="red")

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Scatter Plot with Regression Line')
    plt.legend()

    plt.show()
    


    