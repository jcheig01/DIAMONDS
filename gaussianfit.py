import numpy as np
import matplotlib.pyplot as plt
import torch
import math


def pdf(x, mu, std):

    return (1.0 / (std * math.sqrt(2 * math.pi))) * torch.exp(-0.5 * ((x - mu) / std) ** 2)

def error(x, beta, mu, std):

    f = []
    g = []
    for i in range(len(x)):
        f.append((0.5 * pdf(x[i], -1, 1)) + (0.5 * pdf(x[i], 1, 1)))
        g.append(((torch.exp(beta[0])/(torch.exp(beta[0])+torch.exp(beta[1]))) * pdf(x[i], mu[0], std[0])) + ((torch.exp(beta[1])/(torch.exp(beta[0])+torch.exp(beta[1]))) * pdf(x[i], mu[1], std[1])))

    err = []
    for i in range(len(x)):
        err.append((f[i]-g[i])**2)

    return f, g, [torch.exp(beta[0])/(torch.exp(beta[0])+torch.exp(beta[1])), torch.exp(beta[1])/(torch.exp(beta[0])+torch.exp(beta[1]))], sum(err)/len(err)



def main():
    x = torch.linspace(-1, 5, 100)

    beta = torch.tensor([1, 1], dtype= torch.float, requires_grad=True)
    mu = torch.tensor([-2, 0], dtype= torch.float, requires_grad=True)
    std = torch.tensor([2, 2], dtype= torch.float, requires_grad=True)

    _, initial, _, _ = error(x, beta, mu, std)

    f = []
    g= []

    # Optimizer
    learning_rate = 0.1
    optimizer = torch.optim.Adam([beta, mu, std], lr=learning_rate)

    k = 1000
    for epoch in range(k):
        f, g, alpha, cost = error(x, beta, mu, std)

        # Prevents gradient accumulation
        optimizer.zero_grad()

        # Gradients computed
        cost.backward()

        # Updates parameters
        optimizer.step()

        '''
        with torch.no_grad():
            beta -= learning_rate * beta.grad
            mu -= learning_rate * mu.grad
            std -= learning_rate * std.grad

            # Manually zero the gradients after updating weights
            beta.grad = None
            mu.grad = None
            std.grad = None
        '''



        with torch.no_grad():
            # std greater than 0
            std[:] = std.clamp(0)
            # alpha greater than 0
            beta[:] = beta.clamp(0)


        print('step {}: alpha={:.7f}:{:.7f},mu={}, std={}, cost={}'.format(epoch, alpha[0], alpha[1], mu.tolist(), std.tolist(), cost))


    plt.plot(x, f, 'r', label='f')
    plt.plot(x, initial, 'g', label='initial')
    plt.plot(x, g, 'b', label='final')
    plt.legend()
    plt.show()

    return

if __name__ == '__main__':
    main()