###################################################
# IFT3395 - devoir 3
# Auteurs: Boumedienne Boukharouba, Farzin Faridfar
# 4 decembre 2017
###################################################
import numpy as np
import matplotlib.pyplot as plt
import time
import gzip,pickle

from NeuralNetwork import *
from NeuralNetworkMatrix import *
from math_tools import printDimensions
from plotter import grid_plot, learning_plot

# preparer les donnees de 2 moons
def get_2moons_data():
    deux_moons = np.loadtxt('2moons.txt')
    
    inds = list(range(deux_moons.shape[0]))
    np.random.shuffle(inds)

    # Supposons qu'on fait l'entrainement sur 80% des donnees
    # et le test sur le 20% du reste
    n_train = np.ceil(0.8 * deux_moons.shape[0]).astype(int)
    train_inds = inds[:n_train]
    test_inds = inds[n_train:]

    train_set = deux_moons[train_inds,:]
    test_set = deux_moons[test_inds,:]
    
    # Normaliser les données
    mu1 = train_set[:,0].mean()
    mu2 = train_set[:,1].mean()
    sigma1 = train_set[:,0].std()
    sigma2 = train_set[:,1].std()
    train_set[:,0] -= mu1
    train_set[:,1] -= mu2
    train_set[:,0] /= sigma1
    train_set[:,1] /= sigma2
    test_set[:,0] -= mu1
    test_set[:,1] -= mu2
    test_set[:,0] /= sigma1
    test_set[:,1] /= sigma2
    
    return [train_set, test_set]

# prepare les donnees de MNIST
def get_mnistData():
    f = gzip.open('mnist.pkl.gz')
    # pour resoudre l'erreur d'encoding
    data = pickle.load(f, encoding='latin1')
    # data = pickle.load(f)

    # matrice de train data
    train_inputs = data[0][0]
    # normalizer des donnes
    train_normal = train_inputs/np.array([
        np.sum(train_inputs, axis=1)]).T
    # vecteur des train labels
    train_lables = np.array([data[0][1]]).T
    train_set = np.append(train_normal, train_lables, axis=1)

    # matrice de valid data
    valid_inputs = data[1][0]
    # normalizer des donnes
    valid_normal = valid_inputs/np.array([
        np.sum(valid_inputs, axis=1)]).T
    # vecteur des valid labels
    valid_lables = np.array([data[1][1]]).T
    valid_set = np.append(valid_normal, valid_lables, axis=1)
        
    # matrice de test data
    test_inputs = data[2][0]
    test_mu = np.array([np.mean(test_inputs, axis=1)])
    test_sigma = np.array([np.std(test_inputs, axis=1)])
    # normalizer des donnes
    test_normal = test_inputs/np.array(
        [np.sum(test_inputs, axis=1)]).T
    # vecteur des test labels
    test_lables = np.array([data[2][1]]).T
    test_set = np.append(test_normal, test_lables, axis=1)
    
    return [train_set, valid_set, test_set]


# la methode pour verifier les gradients calcules par la methode
# retropropagation et la methode de difference finie
# (pour repondre aux question 1, 2 et 4)
def gradient_comparison(data, dh, k=1, epochs=1, eta=0.1):
    # creer un model de reseau de neurones pour calculer
    # les gradients avec la methode retropropagation
    model = NeuralNetwork.NeuralNetwork(epochs, dh, k)
    model.train(data, 2)

    # appeler la fonction de difference finie pour calculer
    # le ratio
    ratio_mat = finite_diff(model, data, k)

    param_list = ['grad_w1', 'grad_b1', 'grad_w2', 'grad_b2']

    # l'affichage de resultat pour chacun des params
    for i,ratio in enumerate(ratio_mat):
        print(param_list[i] + ":")
        print(ratio)
        print()

# la methode pour comparer le role des differents params
# (pour repondre a la question 5)
def hyperParam_study(train_set, test_set,
                     n_classes=2, k=100, eta=0.1):
    
    test_inputs = test_set[:,:-1]
    test_labels = test_set[:,-1]

    # lists des different valeurs de chaque param pour obtenir
    # le resultat de leurs effets sur l'erreur de generalisation
    epoch_list = [5, 15, 50]
    dh_list = [2, 5, 10]
    lambda12_list = [0, 0.1, 0.0001]

    for epoch in epoch_list:
        for dh in dh_list:
            for lambda1 in lambda12_list:
                for lambda2 in lambda12_list:
                    print("Les resultat pour "
                          "epoch={epoch}, dh={dh}, lambda1={lambda1} et lambda2={lambda2}".format(
                              epoch=epoch, dh=dh, lambda1=lambda1, lambda2=lambda2))
                    # creer le model de reseaux de neurones
                    model = NeuralNetwork.NeuralNetwork(epoch, dh, k, lambda1, lambda2)
                    model.train(train_set, n_classes)

                    # obtenir les classes pour un ensemble de test
                    classes_pred = model.compute_predictions(test_inputs)

                    # calculer l'erreur de generalisation
                    err = 1.0 - np.mean(test_labels==classes_pred)
                    print("L'erreur de version elementaire"
                          , 100.0 * err,"%")
                    print()

                    # plotter la surface de decision
                    # grid_plot(model, train_set, test_set)


# comparaison des dimension de chacun des params et leurs
# gradients pour deux techniques elementaire et matricielle
# (pour repondre a la question 6)
def dimension_comparison(train_set, test_set, epochs=25,
                     n_classes=2, eta=0.1):
    
    test_inputs = test_set[:,:-1]
    test_labels = test_set[:,-1]
    dh = 2
    k = 10

    # dimension de la matrice de donnees
    x_dim = (k, train_set.shape[1])
    
    # methode elementaire
    print("Methode elementaire sur le lot avec k =", k)
    # creer le model de reseaux de neurones
    model = NeuralNetwork.NeuralNetwork(epochs, dh, k)
    model.train(train_set, n_classes)
    # obtenir les classes pour un ensemble de test
    classes_pred = model.compute_predictions(test_inputs)
    # l'affichage des dimension
    printDimensions(model, x_dim)
    print()
    
    # methode matricielle
    print("Methode matricielle sur le lot avec k =", k)
    # creer le model de reseaux de neurones
    modelMatrix = NeuralNetworkMatrix(epochs, dh, k)
    modelMatrix.train(train_set, n_classes)
    # obtenir les classes pour un ensemble de test
    classes_pred = modelMatrix.compute_predictions(test_inputs)
    # l'affichage des dimension
    printDimensions(modelMatrix, x_dim)
    print()
    print()
    
# comparer les technique de calculs elementaire et matricielle
# sur les batch et obtenir le temps de calcul et
# l'erreur de generalisation
# (pour repondre aux question 7 et 8)
def batch_comparison(train_set, test_set, k_list, epochs=25,
                     n_classes=2, eta=0.1):
    
    test_inputs = test_set[:,:-1]
    test_labels = test_set[:,-1]
    dh = 10

    for k in k_list:
        # methode elementaire
        print("Methode elementaire sur le lot avec k =", k)
        start_time = time.time()
        # creer le model de reseaux de neurones
        model = NeuralNetwork.NeuralNetwork(epochs, dh, k)
        model.train(train_set, n_classes)
        # obtenir les classes pour un ensemble de test
        classes_pred = model.compute_predictions(test_inputs)
        # l'affichage de temps de calcul
        print("Temps de l'execution version elementaire: {time}".format(time=(time.time() - start_time)))
        # l'affichage de l'erreur de generalisation
        err = 1.0 - np.mean(test_labels==classes_pred)
        print("L'erreur de version elementaire ", 100.0 * err,"%")
        print()
        
        # methode matricielle
        print("Methode matricielle sur le lot avec k =", k)
        start_time = time.time()
        # creer le model de reseaux de neurones
        modelMatrix = NeuralNetworkMatrix(epochs, dh, k)
        modelMatrix.train(train_set, n_classes)
        # obtenir les classes pour un ensemble de test
        classes_pred = modelMatrix.compute_predictions(test_inputs)
        # l'affichage de temps de calcul
        print("Temps de l'execution version matricielle: {time}".format(time=(time.time() - start_time)))
        # l'affichage de l'erreur de generalisation
        err = 1.0 - np.mean(test_labels==classes_pred)
        print("L'erreur de version matricielle ", 100.0 * err,"%")
        print()

##### Repondre aux questions de la partie pratique
deux_moons = get_2moons_data()
print("Le ratio des calculs des gradient"
          " avec la method de difference finie par rapport"
              " a la methode de retropropagation:")
print()
data_q1q2 = np.array([deux_moons[0][0]])

print("Questin_1 (d=2, dh=1, k=1):")
gradient_comparison(data_q1q2, 1)
print("============================================================")

print("Questin_2 (d=2, dh=2, k=1):")
gradient_comparison(data_q1q2, 2)
print("============================================================")

data_q4 = np.array(deux_moons[0][0:10])
print("Questin_4 (d=2, dh=2, k=10):")
gradient_comparison(data_q4, 2, 10)

train_set_q5q6q7 = np.array(deux_moons[0])
test_set_q5q6q7 = np.array(deux_moons[1])
print("============================================================")

print("Questin_5 comparaison des hyper-parametres:")
hyperParam_study(train_set_q5q6q7, test_set_q5q6q7)
print("============================================================")

print("Questin_6 comparaison des dimensions des "
      "models elementaire et matricielle:")
dimension_comparison(train_set_q5q6q7, test_set_q5q6q7)
print("============================================================")

print("Questin_7 comparaison des methodes elementaire "
      "et matricielle sur les lots (2moons):")
batch_comparison(train_set_q5q6q7, test_set_q5q6q7, [1,10])
print("============================================================")

train_set_q8 = get_mnistData()[0]
test_set_q8 = np.array(get_mnistData()[2])
print("Questin_8 comparaison des methodes elementaire "
      "et matricielle sur les lots (MNIST):")
batch_comparison(train_set_q8, test_set_q8, [100], 1, 10)
print("============================================================")

# print("Questin_9 calculs au vol pour les ensembles de train, "
#       "test et valid (methode matricielle, MNIST):")
# train_set_q9 = get_mnistData()[0]
# valid_set_q9 = get_mnistData()[1]
# test_set_q9 = get_mnistData()[2]
# test_inputs_q9 = test_set_q9[:,:-1]
# test_labels_q9 = test_set_q9[:,-1]

# epochs = 550
# dh = 25
# k = 200
# lambda1 = 0
# lambda2 = 0

# print("Resultat de calcule pour epochs={epochs}, dh={dh}, k={k}, lambda1={lambda1}, lambda2={lambda2}".format(epochs=epochs, dh=dh, k=k, lambda1=lambda1, lambda2=lambda2))
# modelMatrix = NeuralNetworkMatrix(epochs, dh, k, lambda1, lambda2)
# modelMatrix.study_log(train_set_q9, valid_set_q9, test_set_q9)
# modelMatrix.train(train_set_q9, 10)
# print("============================================================")

print("Questin_10 courbes de taux d’erreurs de classification "
      "courbe de la perte moyenne (methode matricielle, MNIST):")
study_data = np.loadtxt("study_log.txt")

learning_plot(study_data[:,0], study_data[:,1]
              , study_data[:,2], study_data[:,3], "Perte Moyenne")

learning_plot(study_data[:,0], study_data[:,4]
              , study_data[:,5], study_data[:,6], "Erreur Moyenne")
