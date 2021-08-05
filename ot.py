import cProfile

import matplotlib.pyplot as plt
from matplotlib import cm
import torch
import math

def pdf_bi(x, mu, std, ro):

    # 2x2 inverse
    deter = (std[0] ** 2 * std[1] ** 2) - (ro * std[0] * std[1]) ** 2
    inv_sigma = [[(1/deter) * (std[1] ** 2), -(1/deter) * ro * std[0] * std[1]], [-(1/deter) * ro * std[0] * std[1], (1/deter) * (std[0] ** 2)]]

    x = torch.tensor(x)
    inv_sigma = torch.tensor(inv_sigma)

    p1 = 1 / (((2 * math.pi) ** (len(mu) / 2)) * (deter) ** (1 / 2))
    p2 = (-1 / 2) * torch.dot(torch.matmul((x-mu), inv_sigma), (x-mu))

    p = p1 * torch.exp(p2)

    return p

def pdf(x, mu, std, alpha):

    return alpha * ((1.0 / (std * math.sqrt(2 * math.pi))) * torch.exp(-0.5 * ((x - mu) / std) ** 2))

def error(x, y, beta, mu, std, ro, lamb, lamb2):

    alpha = [torch.exp(beta[0])/(torch.exp(beta[0])+torch.exp(beta[1])), torch.exp(beta[1])/(torch.exp(beta[0])+torch.exp(beta[1]))]

    # j1: Optimal Transport
    a = b = c = d = 0

    for i in range(len(mu)):
        a += alpha[i] * (mu[i][0] ** 2 + std[i][0] ** 2)
        b += alpha[i] * (mu[i][0] * mu[i][1] + ro[i] * std[i][0] * std[i][1])
        c += alpha[i] * (mu[i][0] * mu[i][1] + ro[i] * std[i][0] * std[i][1])
        d += alpha[i] * (mu[i][1] ** 2 + std[i][1] ** 2)

    j1 = a - c - b + d

    # j2: Kullback-Libler Distance
    dkl_1 = 0
    dkl_2 = 0
    for i in range(len(mu)):
        for j in range(len(x)):
            dkl_1 += pdf(x[j], -0.3, 0.25, 1)*(torch.log10(pdf(x[j], -0.3, 0.25, 1)) - torch.log10(pdf(x[j], mu[i][0], std[i][0], alpha[i])))
            dkl_2 += pdf(y[j], 0.4, 0.2, 1)*(torch.log10(pdf(y[j], 0.4, 0.2, 1)) - torch.log10(pdf(y[j], mu[i][1], std[i][1], alpha[i])))

    j2 = dkl_1 + dkl_2

    # j3: Andy's cost
    j3 = 0
    for i in range(len(ro)):
        numer = 0
        denom1 = 0
        denom2 = 0
        for j in range(len(x)):
            numer += (x[j] - mu[i][0])*(y[j] - mu[i][1])
            denom1 += (x[j] - mu[i][0])**2
            denom2 += (y[j] - mu[i][1])**2

        j3 += (ro[i] - numer/torch.sqrt(denom1*denom2))**2

    j = j1 + lamb*j2 + lamb2*j3

    return j1, lamb*j2, j, alpha

def main():
    '''
    lamb = torch.logspace(-4,1,20)
    j1_list = []
    j2_list = []
    cost_list = []

    for l in lamb:
        print("Lambda = " + str(l.item()))
        print("-------------------------------------------------")

        x = torch.linspace(-1.5, 1.5, 50)
        y = torch.linspace(-1.5, 1.5, 50)

        beta = torch.tensor([0.5, 0.5], dtype= torch.float, requires_grad=True)
        mu = torch.tensor([[1, 0], [0, 1.2]], dtype= torch.float, requires_grad=True)
        std = torch.tensor([[1, 1.5], [1.5, 1]], dtype= torch.float, requires_grad=True)
        ro = torch.tensor([0.7, 0.8], dtype= torch.float, requires_grad=True)
        # ro = torch.tensor([0, 0], dtype= torch.float)

        # Optimizer
        learning_rate = 0.05
        optimizer = torch.optim.Adam([beta, mu, std, ro], lr=learning_rate)
        # optimizer = torch.optim.Adam([beta, mu, std], lr=learning_rate)

        k = 10
        for epoch in range(k):

            j1, j2, cost, alpha = error(x, y, beta, mu, std, ro, l, 50)

            # Prevents gradient accumulation
            optimizer.zero_grad()

            # Gradients computed
            cost.backward()

            # Updates parameters
            optimizer.step()

            with torch.no_grad():
                # std greater than 0
                std[:] = std.clamp(0, 0.5)
                # alpha greater than 0
                beta[:] = beta.clamp(0)
                # ro between -1 and 1
                ro[:] = ro.clamp(-1, 1)

            print('step {}: alpha={:.5f}:{:.5f},mu={}, std={}, ro={}, cost={}'.format(epoch, alpha[0], alpha[1], mu.tolist(), std.tolist(), ro.tolist(), cost))
            # print('step {}: alpha={:.5f}:{:.5f},mu={}, std={}, cost={}'.format(epoch, alpha[0], alpha[1], mu.tolist(), std.tolist(), cost))


        j1_list.append(j1)
        j2_list.append(j2)
        cost_list.append(cost)

    plt.plot(lamb, cost_list, 'r', label='cost')
    plt.plot(lamb, j1_list, 'g', label='j1')
    plt.plot(lamb, j2_list, 'b', label='j2')
    plt.yscale("log")
    plt.legend()
    plt.show()
    '''

    # w/ best lambda
    x = torch.linspace(-1.5, 1.5, 50)
    y = torch.linspace(-1.5, 1.5, 50)

    beta = torch.tensor([0.5, 0.5], dtype=torch.float, requires_grad=True)
    mu = torch.tensor([[1, 0], [0, 1.2]], dtype=torch.float, requires_grad=True)
    std = torch.tensor([[1, 1.5], [1.5, 1]], dtype=torch.float, requires_grad=True)
    ro = torch.tensor([0.7, 0.8], dtype= torch.float, requires_grad=True)
    # ro = torch.tensor([0, 0], dtype=torch.float)

    # Optimizer
    learning_rate = 0.05
    optimizer = torch.optim.Adam([beta, mu, std, ro], lr=learning_rate)
    # optimizer = torch.optim.Adam([beta, mu, std], lr=learning_rate)

    k = 150
    for epoch in range(k):
        j1, j2, cost, alpha = error(x, y, beta, mu, std, ro, 0.033, 55)

        # Prevents gradient accumulation
        optimizer.zero_grad()

        # Gradients computed
        cost.backward()

        # Updates parameters
        optimizer.step()

        with torch.no_grad():
            # std greater than 0
            std[:] = std.clamp(0, 0.5)
            # alpha greater than 0
            beta[:] = beta.clamp(0)
            # ro between -1 and 1
            ro[:] = ro.clamp(-1, 1)

        print('step {}: alpha={:.5f}:{:.5f},mu={}, std={}, ro={}, cost={}'.format(epoch, alpha[0], alpha[1], mu.tolist(), std.tolist(), ro.tolist(), cost))
        # print('step {}: alpha={:.5f}:{:.5f},mu={}, std={}, cost={}'.format(epoch, alpha[0], alpha[1], mu.tolist(), std.tolist(), cost))

    # Marginal Fit
    y1 = []
    fit1 = []
    y2 = []
    fit2 = []

    for i in range(len(x)):
        y1.append(pdf(x[i], -0.3, 0.25, 1))
        fit1.append(pdf(x[i], mu[0][0], std[0][0], alpha[0]) + pdf(x[i], mu[1][0], std[1][0], alpha[1]))
        y2.append(pdf(x[i], 0.4, 0.2, 1))
        fit2.append(pdf(y[i], mu[0][1], std[0][1], alpha[0]) + pdf(y[i], mu[1][1], std[1][1], alpha[1]))
    
    plt.subplot(2, 1, 1)
    plt.plot(x, y1, 'r', label='true x')
    plt.plot(x, fit1, 'g', label='fit x')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(x, y2, 'r', label='true y')
    plt.plot(x, fit2, 'g', label='fit y')
    plt.legend()
    plt.show()

    X, Y = torch.meshgrid(x,y)
    fit_2d = torch.zeros(X.shape)
    # 2D Fit
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            fit_2d[i, j] = alpha[0]*pdf_bi([X[i, j], Y[i, j]], mu[0], std[0], ro[0]) \
                           + alpha[1]*pdf_bi([X[i, j], Y[i, j]], mu[1], std[1], ro[1])

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X.numpy(), Y.numpy(), fit_2d.detach().numpy(), cmap=cm.coolwarm, linewidth=0, antialiased=False)

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

    return

if __name__ == '__main__':
    main()