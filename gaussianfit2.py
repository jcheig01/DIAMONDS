import matplotlib.pyplot as plt
import torch
import math


def pdf(x, mu, std):

    return (1.0 / (std * math.sqrt(2 * math.pi))) * torch.exp(-0.5 * ((x - mu) / std) ** 2)

def error(x, beta, mu, std, gaussian):

    # Calculate alpha
    beta_sum = 0
    for i in range(gaussian):
        beta_sum += torch.exp(beta[i])

    alpha = [torch.exp(beta[i])/beta_sum for i in range(gaussian)]

    # Get function f values & fit g values
    f = []
    g = []
    for i in range(len(x)):
        f.append(1/(x[-1]-x[0]))

        fit = 0

        for j in range(gaussian):
            fit += alpha[j] * pdf(x[i], mu[j], std[j])

        g.append(fit)

    # MSE
    err = [(f[i]-g[i])**2 for i in range(len(x))]

    return f, g, alpha, sum(err)/len(err)


def main():
    start = -6
    end = 10
    n = 100

    x = torch.linspace(start, end, n)
    delta = float(end-start)/n

    pre_beta = [1, 1, 3, 5, -1, 2, 1, 1, 3, 5, -1, 2, 1, 1, 3, 5, -1, 2, 1, 1]
    pre_mu = [-1, 1, 3, 5, -1, 2, -1, 1, 3, 5, -1, 2, -1, 1, 3, 5, -1, 2, -1, 1]
    pre_std = [2, 2, 1, 1.2, 2.4, 1.5, 2, 2, 1, 1.2, 2.4, 1.5, 2, 2, 1, 1.2, 2.4, 1.5, 2, 2, 1, 1.2, 2.4, 1.5, 2, 2]

    maxGaussian = len(pre_beta)

    f = []
    g_list= []
    cost_list = []
    alpha_list = []
    mu_list = []
    std_list = []

    # Optimizer
    learning_rate = 0.1

    k = 600

    for gaussian in range(1,maxGaussian+1):

        beta = torch.tensor(pre_beta[:gaussian], dtype=torch.float, requires_grad=True)
        mu = torch.tensor(pre_mu[:gaussian], dtype=torch.float, requires_grad=True)
        std = torch.tensor(pre_std[:gaussian], dtype=torch.float, requires_grad=True)

        optimizer = torch.optim.Adam([beta, mu, std], lr=learning_rate)

        print("NUMBER OF GAUSSIAN: " + str(gaussian))
        print("-------------------------")
        for epoch in range(k):
            f, g, alpha, cost = error(x, beta, mu, std, gaussian)

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

            # For printing purpose
            alpha_val = [a.item() for a in alpha]

            print('step {}: alpha={},mu={}, std={}, cost={}'.format(epoch, alpha_val, mu.tolist(), std.tolist(), cost))

            if epoch == k-1:
                g_list.append(g)
                alpha_list.append(alpha_val)
                mu_list.append(mu)
                std_list.append(std)
                cost_list.append(cost)

        riemann = sum(g) * delta
        print("Riemann Sum: {}".format(riemann))

    print("-------------------------")
    for i in range(maxGaussian):
        print('Num Gaussian={}: alpha={},mu={}, std={}, cost={}'.format(i+1, alpha_list[i], mu_list[i].tolist(), std_list[i].tolist(), cost_list[i]))
        print("-------------------------")


    # Subplots

    plt.figure()

    for i in range(20):
        ax = plt.subplot(5, 4, i+1)
        plt.plot(x, f, 'r', label='f')
        plt.plot(x, g_list[i], 'b', label='final')
        plt.xlim([-8, 12])
        plt.ylim([-0.05, 0.2])
        ax.set_title('Cost: {:.5f}'.format(cost_list[i]))

    plt.show()

    x = range(1, maxGaussian+1)

    plt.figure()
    plt.plot(x, cost_list)
    plt.show()


    # Sanity check with riemann sum


if __name__ == '__main__':
    main()