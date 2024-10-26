import torch 
import matplotlib.pyplot as plt

"""
Create an 8 by 5 tensor initialized randomly with floats chosen from normal distributions with mean, 
in turn, equal to 1, 2,...,40, and standard deviation equal to 4. Hint: see Random Sampling; specifically, 
torch.normal().
"""

print(torch.normal(torch.FloatTensor([i+1 for i in range(40)]),5).view(8, 5))


if __name__ == "__main__":

    print(torch.randn(100000).mean())

    xs = 100 * torch.rand(40)
    ys = torch.normal(2 * xs + 9, 20)

    plt.scatter(xs.numpy(), ys.numpy())
    plt.show()
