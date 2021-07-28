import cProfile

import matplotlib.pyplot as plt
from matplotlib import cm
import torch
import math

'''
def matmul (A, B):
    rows_A = len(A)
    cols_A = len(A[0])
    rows_B = len(B)
    cols_B = len(B[0])

    if cols_A != rows_B:
      print("Cannot multiply the two matrices. Incorrect dimensions.")
      return

    # Create the result matrix
    # Dimensions would be rows_A x cols_B
    C = [[0 for row in range(cols_B)] for col in range(rows_A)]
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                C[i][j] += A[i][k] * B[k][j]
    return C
'''

def pdf_bi(x, mu, std, ro):
    '''
    # Bivariate

    # 2x2 inverse
    deter = (std[0] ** 2 * std[1] ** 2) - (ro * std[0] * std[1]) ** 2
    inv_sigma = [[deter * (std[1] ** 2), -deter * ro * std[0] * std[1]], [-deter * ro * std[0] * std[1], deter * (std[0] ** 2)]]

    x_mu = []
    x_mu_T = []
    for i in range(len(x)):
        x_mu.append(x[i]-mu[i])
        x_mu_T.append([x[i]-mu[i]])

    p1 = 1 / (((2 * math.pi) ** (len(mu) / 2)) * (deter) ** (1 / 2))
    p2 = (-1 / 2) * matmul(matmul([x_mu], (inv_sigma)), x_mu_T)[0][0]
    '''
    # Reduced by 11.1s for k=5 (curr: 22.72s)
    # 2x2 inverse
    deter = (std[0] ** 2 * std[1] ** 2) - (ro * std[0] * std[1]) ** 2
    inv_sigma = [[deter * (std[1] ** 2), -deter * ro * std[0] * std[1]], [-deter * ro * std[0] * std[1], deter * (std[0] ** 2)]]

    x = torch.tensor(x)
    inv_sigma = torch.tensor(inv_sigma)

    p1 = 1 / (((2 * math.pi) ** (len(mu) / 2)) * (deter) ** (1 / 2))
    p2 = (-1 / 2) * torch.dot(torch.matmul((x-mu), inv_sigma), (x-mu))

    p = p1 * torch.exp(p2)

    return p

def error(x, y, beta, mu, std, ro):

    alpha = [torch.exp(beta[0])/(torch.exp(beta[0])+torch.exp(beta[1])), torch.exp(beta[1])/(torch.exp(beta[0])+torch.exp(beta[1]))]

    f = torch.zeros(x.shape)
    g = torch.zeros(x.shape)

    err = []
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            f[i, j] = 0.5*pdf_bi([x[i, j], y[i, j]], torch.tensor([2, 1]), [1, 1], 0.9) + 0.5*pdf_bi([x[i, j], y[i, j]], torch.tensor([1, 2]), [1, 1], 0.9)

            g[i, j] = (alpha[0]) * pdf_bi([x[i, j], y[i, j]], mu[0], std[0], ro) \
                    + (alpha[1]) * pdf_bi([x[i, j], y[i, j]], mu[1], std[1], ro)

            err.append((f[i, j] - g[i, j]) ** 2)

    return f, g, alpha, sum(err)/len(err)

def main():
    x = torch.linspace(-8, 8, 50)
    y = torch.linspace(-8, 8, 50)

    x, y = torch.meshgrid(x,y)

    beta = torch.tensor([0.5, 0.5], dtype= torch.float, requires_grad=True)
    mu = torch.tensor([[2, 3], [2,2]], dtype= torch.float, requires_grad=True)
    std = torch.tensor([[1, 1.5], [1.5, 1]], dtype= torch.float, requires_grad=True)
    ro = torch.tensor([0.7], dtype= torch.float, requires_grad=True)

    _, initial, _, _ = error(x, y, beta, mu, std, ro)

    f = torch.zeros(x.shape)
    g = torch.zeros(x.shape)

    # Optimizer
    learning_rate = 0.05
    optimizer = torch.optim.Adam([beta, mu, std, ro], lr=learning_rate)

    k = 5
    for epoch in range(k):

        f, g, alpha, cost = error(x, y, beta, mu, std, ro)

        # Prevents gradient accumulation
        optimizer.zero_grad()

        # Gradients computed
        cost.backward()

        # Updates parameters
        optimizer.step()

        with torch.no_grad():
            # std greater than 0
            std[:] = std.clamp(0)
            # alpha greater than 0
            beta[:] = beta.clamp(0)
            # ro between -1 and 1
            ro[:] = ro.clamp(-1, 1)

        print('step {}: alpha={:.5f}:{:.5f},mu={}, std={}, ro={}, cost={}'.format(epoch, alpha[0], alpha[1], mu.tolist(), std.tolist(), ro.tolist(), cost))


    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(x.numpy(), y.numpy(), f.numpy(), cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.plot_surface(x.numpy(), y.numpy(), g.detach().numpy(), cmap=cm.binary, linewidth=0, antialiased=False)

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


    return

if __name__ == '__main__':
    main()