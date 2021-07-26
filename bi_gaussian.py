import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mayavi import mlab

def prob_density_bi(x):

        # Bivariate
        x = np.array(x)
        mu = np.array([0, 0])

        std1 = 3
        std2 = 3
        ro = 0.3
        sigma = np.array([[std1**2, ro*std1*std2],[ro*std1*std2, std2**2]])

        p1 = 1/(((2*np.pi)**(len(mu)/2))*(np.linalg.det(sigma))**(1/2))
        p2 = (-1/2)*np.matmul(np.matmul((np.transpose(x-mu)),(np.linalg.inv(sigma))),(x-mu))

        p = p1 * np.exp(p2)

        return p

def prob_density_tri(x):
        x = np.array(x)
        mu = np.array([2, 0, 2])

        std1 = 1.0
        std2 = 1.0
        std3 = 1.0

        ro1 = 0.9
        ro2 = 0.9
        ro3 = 0.9
        sigma = np.array([[std1**2, ro1*std1*std2, ro2*std1*std3],[ro1*std1*std2, std2**2, ro3*std2*std3],
                         [ro2*std1*std3, ro3*std2*std3, std3**2]])

        p1 = 1/(((2*np.pi)**(len(mu)/2))*(np.linalg.det(sigma))**(1/2))
        p2 = (-1/2)*np.matmul(np.matmul((np.transpose(x-mu)),(np.linalg.inv(sigma))),(x-mu))

        p = p1 * np.exp(p2)

        return p

def main():
        '''
        # Bivariate
        X = np.arange(-5, 5, 0.25)
        Y = np.arange(-5, 5, 0.25)
        X, Y = np.meshgrid(X, Y)

        pdf = np.zeros(X.shape)
        for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                        pdf[i, j] = prob_density_bi([X[i, j], Y[i, j]])

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(X, Y, pdf, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)

        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()
        '''

        # Trivariate
        X, Y, Z = np.mgrid[-5:5:50j, -5:5:50j, -5:5:50j]
        
        pdf = np.zeros(X.shape)
        for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                        for k in range(X.shape[2]):
                                pdf[i, j, k] = prob_density_tri([X[i, j, k], Y[i, j, k], Z[i, j, k]])
        
        mlab.contour3d(X, Y, Z, pdf, opacity=0.5)
        mlab.axes()
        mlab.show()


        '''
        from scipy import stats

        mu = np.array([1, 10, 20])
        sigma = np.matrix([[20, 10, 10],
                           [10, 25, 1],
                           [10, 1, 50]])
        np.random.seed(100)
        data = np.random.multivariate_normal(mu, sigma, 1000)
        values = data.T

        kde = stats.gaussian_kde(values)

        # Create a regular 3D grid with 50 points in each dimension
        xmin, ymin, zmin = data.min(axis=0)
        xmax, ymax, zmax = data.max(axis=0)
        xi, yi, zi = np.mgrid[xmin:xmax:50j, ymin:ymax:50j, zmin:zmax:50j]

        # Evaluate the KDE on a regular grid...
        coords = np.vstack([item.ravel() for item in [xi, yi, zi]])
        density = kde(coords).reshape(xi.shape)

        mlab.contour3d(xi, yi, zi, density, opacity=0.5)
        mlab.axes()
        mlab.show()
        '''





if __name__ == '__main__':
    main()