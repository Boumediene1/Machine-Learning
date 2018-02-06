import numpy as np
import matplotlib.pyplot as pylab

from gauss_mv import *
from parzen_mv import *

# Ouvrir le fichier iris.txt
iris=np.loadtxt('iris.txt')

# Creer les indices aleatoires
indices_c1 = np.arange(0,50)

# Creer l'ensemble d'entraiment de la classe 1
iris_train_c1 = iris[indices_c1]

################ Une dimension ################
# L'ensemble d'entrainement d'une dimension
iris_train_1d = iris_train_c1[:,0]
iris_train_1d.sort()

# Model Gaussien
gauss = gauss_mv(1)
gauss.train(iris_train_1d)
prob_gauss = np.exp(gauss.compute_predictions(iris_train_1d))
print("Sigma Carre Gaussien Iso = ", gauss.sigma_sq)

# Model Parzen sigma approprie
parzen = parzen_mv(1, 0.1)
parzen.train(iris_train_1d)
prob_parzen = np.exp(parzen.compute_predictions(iris_train_1d))

# Model Parzen sigma petit
parzen_sigma_p = parzen_mv(1, 0.02)
parzen_sigma_p.train(iris_train_1d)
prob_parzen_sigma_p = np.exp(parzen_sigma_p.compute_predictions(iris_train_1d))

# Model Parzen sigma grand
parzen_sigma_g = parzen_mv(1, 0.8)
parzen_sigma_g.train(iris_train_1d)
prob_parzen_sigma_g = np.exp(parzen_sigma_g.compute_predictions(iris_train_1d))


### Affichage 1D
# pylab.figure("1D")
pylab.scatter(iris_train_1d, np.zeros(iris_train_1d.shape[0]))
pylab.plot(iris_train_1d,prob_gauss, label="Guassien")
pylab.plot(iris_train_1d,prob_parzen, label="moyen")
pylab.plot(iris_train_1d,prob_parzen_sigma_p, label="petit")
pylab.plot(iris_train_1d,prob_parzen_sigma_g, label="grand")

pylab.xlabel('x')
pylab.ylabel('Log Probability Parzen')
pylab.legend(('Gaussien','Sigma Moyen', 'Sigma Petit', 'Sigma Grand', 'Points Entrainements'))
pylab.title('1D Gaussien et Parzen')


# ################ Deux dimensions ################
# # L'ensemble d'entrainement d'une dimension
iris_train_2d = iris_train_c1[:,0:2]

n_points = 50

prob_gauss_diag = np.ndarray((n_points, n_points))
prob_parzen_2d = np.ndarray((n_points, n_points))
prob_parzen_2d_sigma_p = np.ndarray((n_points, n_points))
prob_parzen_2d_sigma_g  = np.ndarray((n_points, n_points))

x_points = np.linspace( min(iris_train_2d[:,0]), max(iris_train_2d[:,0]), n_points)
y_points = np.linspace( min(iris_train_2d[:,1]), max(iris_train_2d[:,1]), n_points)

# Model Gaussien
gauss_diag = gauss_mv(2)
gauss_diag.train(iris_train_2d)
print("Sigma Carre Gaussien Diagonal = ", gauss_diag.sigma_sq)

# Model Parzen sigma approprie
parzen_2d = parzen_mv(2, 0.1)
parzen_2d.train(iris_train_2d)

# Model Parzen sigma petit
parzen_2d_sigma_p = parzen_mv(2, 0.05)
parzen_2d_sigma_p.train(iris_train_2d)

# Model Parzen sigma grand
parzen_2d_sigma_g = parzen_mv(2, 1)
parzen_2d_sigma_g.train(iris_train_2d)


for i in range(n_points):
    test_data = np.transpose([np.ones(n_points)*x_points[i], y_points])
    prob_gauss_diag[i] = np.exp(gauss_diag.compute_predictions(test_data))
    prob_parzen_2d[i] = np.exp(parzen_2d.compute_predictions(test_data))
    prob_parzen_2d_sigma_p[i] = np.exp(parzen_2d_sigma_p.compute_predictions(test_data))
    prob_parzen_2d_sigma_g[i] = np.exp(parzen_2d_sigma_g.compute_predictions(test_data))


### Affichage 2D
pylab.figure("Gaussien")
pylab.scatter(iris_train_c1[:,0],iris_train_c1[:,1])
g_plot = pylab.contour(x_points,y_points,np.transpose(prob_gauss_diag))
pylab.figure("Parzen Sigma Moyen")
pylab.scatter(iris_train_c1[:,0],iris_train_c1[:,1])
p_plot = pylab.contour(x_points,y_points,np.transpose(prob_parzen_2d))
pylab.figure("Parzen Sigma Petit")
pylab.scatter(iris_train_c1[:,0],iris_train_c1[:,1])
pp_plot = pylab.contour(x_points,y_points,np.transpose(prob_parzen_2d_sigma_p))
pylab.figure("Parzen Sigma Grand")
pylab.scatter(iris_train_c1[:,0],iris_train_c1[:,1])
pg_plot = pylab.contour(x_points,y_points, np.transpose(prob_parzen_2d_sigma_g))


pylab.show()
