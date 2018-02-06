###################################################
# IFT3395 - devoir 3
# Auteurs: Boumedienne Boukharouba, Farzin Faridfar
# 4 decembre 2017
###################################################
import numpy as np
from math_tools import (
    creat_batch, linearize, rect, softmax, sign, onehot, finite_diff)

class NeuralNetwork():
    def __init__(self, epochs, dh, k=10, lambda1=0.001, lambda2=0.001, eta=0.1):
        self.eta = eta
        self.epochs = epochs
        self.dh = dh
        self.k = k
        # lambda1 et 2: weight decay
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def train(self, train_data, m=2):
        # Pour traiter les vecteurs et les matrices
        # on ajoute une dimension au vecteurs pour unifier
        # l'operation mathematique de numpy
        if train_data.ndim == 1:
            train_data = np.array([train_data])
        
        self.d = train_data.shape[1]-1
        self.n = train_data.shape[0]
        self.m = m
        
        # separer les traits et les cibles des donnees
        x_set = train_data[:, :-1]
        y_set = (train_data[:, -1]).astype(int)
                
        # initialization des params
        self.w1, self.b1 = self.initParam(self.k, (self.dh, self.d))
        self.w2, self.b2 = self.initParam(self.k, (m, self.dh))
        # nombre de lot selon la taille de lot recue
        n_batch = self.n // self.k

        # entrainement de model pour chaque epoch
        for n_iter in range(self.epochs):
            # produir les donnees aleatoirs pour chaque epoch
            inds = list(range(train_data.shape[0]))
            np.random.shuffle(inds)
            train_epoch = train_data[inds,:]
            
            # Creer la matrice qui contient les sous-matrice des batches
            batch_mat = creat_batch(train_epoch, n_batch, self.d+1)
            
            # Pour chaque batch on fait le calcule de fprop et bprop
            # et prendre la moyenne des gradients et
            # finalement mettre a jour les params
            for i,batch in enumerate(batch_mat):
                batch_size = batch.shape[0]
                # Creer les matrices qui contiennent les sous-matrices de
                # gradient de chaque batch
                batch_grw1 = np.ones((batch_size, self.dh, self.d))
                batch_grb1 = np.ones((batch_size, self.dh, 1))
                batch_grw2 = np.ones((batch_size, self.m, self.dh))
                batch_grb2 = np.ones((batch_size, self.m, 1))

                # Separer les traits et les cible de chaque batch
                x_set = batch[:,:-1]
                y_set = (batch[:,-1]).astype(int)

                # Calculs de fprop et bprob pour chacun des exemple dans
                # une batch
                for j in range(batch_size):
                    x = np.array([x_set[j]])
                    y = y_set[j]
                    self.fprop(x)
                    self.bprop(x, y)
                    batch_grw1[j] = self.grad_w1
                    batch_grb1[j] = self.grad_b1
                    batch_grw2[j] = self.grad_w2
                    batch_grb2[j] = self.grad_b2
            

                # Calculer les moyennes des gradients pour chq param
                self.grad_w1 = np.mean(batch_grw1, axis=0)
                self.grad_b1 = np.mean(batch_grb1, axis=0)
                self.grad_w2 = np.mean(batch_grw2, axis=0)
                self.grad_b2 = np.mean(batch_grb2, axis=0)

                # mettre a jour des params apres les calculs de
                # de chaque batch
                self.updateParams()
            
            
    # Initializer chaque param en prenant le nombre de donnees
    # de son entree ainsi que sa dimension
    def initParam(self, n_data, dimension):
        low = -np.sqrt(n_data)
        high = np.sqrt(n_data)
        weights = np.random.uniform(low, high, dimension)
        bias = np.zeros((dimension[0],1))
        return [weights, bias]

    # Mettre a jour les params avec la methode de descent de gradient
    def updateParams(self):
         self.w1 -= self.eta * self.grad_w1
         self.b1 -= self.eta * self.grad_b1
         self.w2 -= self.eta * self.grad_w2
         self.b2 -= self.eta * self.grad_b2

    # cette method calcule les parametre du model
    def fprop(self, x):
        ## Couche 1
        # preactivation, weight decay
        self.ha = linearize(x.T, self.w1, self.b1)
        # activation, RELU
        self.hs = rect(self.ha)
        ## Couche 2
        # preactivation, weight decay
        self.oa = linearize(self.hs, self.w2, self.b2)
        # activation, softmax
        self.os = softmax(self.oa)
        
    # cette methode calcule les gradient de parametre
    def bprop(self, x, y):
        self.grad_oa = self.os - onehot(y, self.m)
        self.grad_w2 = np.dot(self.grad_oa, self.hs.T) + self.lambda1*sign(self.w2) + 2*self.lambda2*self.w2
        self.grad_b2 = self.grad_oa
        self.grad_hs = np.dot(self.w2.T, self.grad_oa)
        self.grad_ha = np.where(self.ha > 0, 1 , 0)*self.grad_hs
        self.grad_w1 = np.dot(self.grad_ha, x) + self.lambda1*sign(self.w1) + 2*self.lambda2*self.w1
        self.grad_b1 = self.grad_ha

    # prevoir le cible selon la perte calculee pour chaque exemple
    def predict_class(self):
        loss = -np.log(self.os)
        c = np.argmin(loss)
        return c

    # retourner le vecteur des cible pour la matrice test
    def compute_predictions(self, test_data):
        # vecteur des cibles prevus
        y_test = []
        for test in test_data:
            x = np.array([test])
            self.fprop(x)
            y_test.append(self.predict_class())
            
        return y_test
