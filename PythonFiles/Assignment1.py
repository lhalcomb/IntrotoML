import torch

xs = torch.randn(30)
xsStd = xs.std()
xsMean = xs.mean()
xss = torch.empty(30).normal_(mean=100, std = 25)

def generateMeanArrayFromSamples(n):
    meanArray = torch.empty(n)

    for i in range(n):
        NormSample = torch.empty(30).normal_(mean = 100, std = 25)
        iterativeMeanOfSample = NormSample.mean()
        meanArray[i] = iterativeMeanOfSample

    overall_mean = meanArray.mean()
    print(f'The overall mean is {overall_mean}')

    return meanArray

def generateStdArrayFromSamples(n):
    stdArray = torch.empty(n)

    for i in range(n):
        NormSample = torch.empty(30).normal_(mean = 100, std = 25)
        iterativeStdOfSample = NormSample.std()
        stdArray[i] = iterativeStdOfSample

    overall_std = stdArray.mean()
    print(f'The overall mean of std is {overall_std}')

    return stdArray

def genUniformMeanArrayFromSamples(n):
    meanArray = torch.empty(n)

    for i in range(n):
        NormSample = torch.empty(30).uniform_(100,125)
        iterativeMeanOfSample = NormSample.mean()
        meanArray[i] = iterativeMeanOfSample
    overall_mean = meanArray.mean()

    print(f'The overall mean of the uniform means is {overall_mean}')

def genUniformStdFromSamples(n):
    stdArray = torch.empty(n)

    for i in range(n):
        NormSample = torch.empty(30).uniform_(100, 125)
        iterativeStdOfSample = NormSample.std()
        stdArray[i] = iterativeStdOfSample
    overall_mean = stdArray.mean()

    print(f'The overall mean of the uniform std is {overall_mean}')



if __name__ == "__main__":
    """ randTensor = torch.Tensor(3,4).random_(3,21)
    print(randTensor)
    longRandTensor = torch.LongTensor(3,4).random_(3,21)
    print(longRandTensor)
    xss = torch.ones(3,4,3)
    print(7 * xss)
    print(xss.shape, xss.dim()) """
    print(f'Normal Distribution Random Number Generation of xs: \n {xs}')
    print(f'The standard deviation of xs: {xsStd}')
    print(f'The mean of xs: {xsMean}')

    print(f'This is the answer to the third problem: \n {xss}')
    print(f'The mean to the third: {xss.mean()}')
    print(f'The std to the third: {xss.std()}')
   #print(torch.ones(3,4) / torch.Tensor([1, 2, 3, 4]).unsqueeze(0))
    """ x = torch.ones(5,2)
    x[:,1] = torch.Tensor([0,1,2,3,4])  # Pythonic slicing, remember that Python has zero indexing  
    print(x) """
    print(torch.empty(30).uniform_(0,1))
    generateMeanArrayFromSamples(500)
    generateStdArrayFromSamples(30)
    genUniformMeanArrayFromSamples(30)
    genUniformStdFromSamples(31)