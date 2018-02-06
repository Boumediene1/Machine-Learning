###################################################
# IFT3395 - devoir 3
# Auteurs: Boumedienne Boukharouba, Farzin Faridfar
# 4 decembre 2017
###################################################
import numpy as np
from math_tools import creat_batch
import NeuralNetwork

class NeuralNetworkMatrix(NeuralNetwork.NeuralNetwork):
    def __init__(self, epochs, dh, k=10,
                 lambda1=0, lambda2=0, eta=0.1):
        # initialiser la classe parent (NeuralNetwork)
        super().__init__(epochs, dh, k=10,
                         lambda1=0, lambda2=0, eta=0.1)
        
        # study_mode est pour le but d'enregistrer les erreurs
        # et les pertes pour les ensemble de train, valid et test
        # pour chaque 50 lot (pour alegrir la quantite de calcul)
        self.study_mode = False

    # si le study_mode est true, on initialise les valeurs d'ensemble
    # train, valid et test
    def study_log(self, train_set, valid_set, test_set):
        # mettre le mode en vrai
        self.study_mode = True

        # separer les traits et les cibles de chacun des ensemble
        self.xTrain = train_set[:, :-1]
        self.yTrain = (train_set[:, -1]).astype(int)

        self.xValid = valid_set[:, :-1]
        self.yValid = (valid_set[:, -1]).astype(int)

        self.xTest = test_set[:, :-1]
        self.yTest = (test_set[:, -1]).astype(int)

    def train(self, train_data, m):
        # Pour traiter les vecteurs et les matrices
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

        # si study_mode est vrai, creer le fichier des logs
        if self.study_mode:
            lof_file = open("study_log.txt", "w")

        # le nombre des lot selon sa taille recue
        n_batch = self.n // self.k
        
        # entrainement de model pour chaque epoch
        for epoch in range(self.epochs):
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
                
                # Separer les traits et les cible de chaque batch
                x_set = batch[:,:-1]
                y_set = (batch[:,-1]).astype(int)

                # Calculs de fprop et bprob pour chacun des
                # exemple dans une batch
                self.fprop(x_set)
                self.bprop(x_set, y_set)
                
                # mettre a jour des params apres les calculs de
                # de chaque batch
                self.updateParams()

            # si study mode est vrai, calculer les pertes et
            # les erreurs pour chaque 50 epoch
            if self.study_mode and (epoch%50 == 0):
                # erreur et perte pour l'ensemble de train
                train_lossTotal = 0
                train_errTotal = 0
                trainPred = self.compute_predictions(self.xTrain)
                trainLoss = self.compute_loss(self.xTrain, self.yTrain)
                train_lossTotal = sum(trainLoss)/len(trainLoss)
                train_errTotal = self.compute_err(self.yTrain, trainPred)
                # erreur et perte pour l'ensemble de valid
                valid_lossTotal = 0
                valid_errTotal = 0
                validPred = self.compute_predictions(self.xValid)
                validLoss = self.compute_loss(self.xValid, self.yValid)
                valid_lossTotal = sum(validLoss)/len(validLoss)
                valid_errTotal = self.compute_err(self.yValid, validPred)
                
                # erreur et perte pour l'ensemble de test
                train_lossTotal = 0
                test_errTotal = 0
                testPred = self.compute_predictions(self.xTest)
                testLoss = self.compute_loss(self.xTest, self.yTest)
                test_lossTotal = sum(testLoss)/len(testLoss)
                test_errTotal = self.compute_err(self.yTest, testPred)
                # l'affichage des resultats sur le terminal
                print("epoch={epoch}".format(epoch=epoch))
                print("train_lossTotal={0:.4f}, train_errTotal={1:.2f}%"
                      .format(train_lossTotal, train_errTotal*100))
                print("valid_lossTotal={0:.4f}, valid_errTotal={1:.2f}%"
                      .format(valid_lossTotal, valid_errTotal*100))
                print("test_lossTotal={0:.4f}, test_errTotal={1:.2f}%".
                      format(test_lossTotal, test_errTotal*100))
                print()

                
                # enregistrer les errerus et les pertes
                # dans le fichier des logs
                lof_file.write(
                    "{0} {1} {2} {3} {4} {5} {6}\n".
                    format(epoch,
                           train_lossTotal, valid_lossTotal, test_lossTotal, train_errTotal, valid_errTotal, test_errTotal))
    
    # Initializer chaque param en prenant le nombre de donnees
    # de son entree ainsi que sa dimension
    def initParam(self, n_data, dimension):
        low = -np.sqrt(n_data)
        high = np.sqrt(n_data)
        weights = np.random.uniform(low, high, dimension)
        bias = np.zeros((dimension[0],n_data))
        return [weights, bias]

    # retourner le vecteur des loss pour l'ensemble de donnees
    def compute_loss(self, x_set, y_set):
        loss_list = []
        for i,x in enumerate(x_set):
            x = np.array([x])
            self.fprop(x)
            loss = -np.log(self.os.T[0])
            loss_list.append(loss[y_set[i]])
        return loss_list

    # prevoir le cible selon la perte calculee pour chaque exemple
    def predict_class(self):
        loss = -np.log(self.os.T[0])
        c = np.argmin(loss)
        return c

    # calculer l'erreur moyenne pour le vecteur des cibles prevus
    def compute_err(self, labels, pred):
        err = 1.0 - np.mean(labels==pred)
        return err

    # retourner le vecteur des cible pour la matrice test
    def compute_predictions(self, test_inputs):
        # vecteur des cibles prevus
        class_list = []
        for test in test_inputs:
            x = np.array([test])
            self.fprop(x)
            y_pred = self.predict_class()
            class_list.append(y_pred)
            
        return class_list
