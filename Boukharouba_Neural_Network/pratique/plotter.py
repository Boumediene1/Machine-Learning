# coding=utf-8
###################################################
# IFT3395 - devoir 3
# Auteurs: Boumedienne Boukharouba, Farzin Faridfar
# 4 decembre 2017
# Reference: code utilitaire.py du seance demo
###################################################

import numpy
import pylab

# fonction plot pour la surface de decision
def grid_plot(NN, train, test, n_points=50):

    train_test = numpy.vstack((train,test))
    (min_x1,max_x1) = (min(train_test[:,0]),max(train_test[:,0]))
    (min_x2,max_x2) = (min(train_test[:,1]),max(train_test[:,1]))

    xgrid = numpy.linspace(min_x1,max_x1,num=n_points)
    ygrid = numpy.linspace(min_x2,max_x2,num=n_points)

	# calcule le produit cartesien entre deux listes
    # et met les resultats dans un array
    thegrid = numpy.array(combine(xgrid,ygrid))

    classesPred = NN.compute_predictions(thegrid)

    # La grille
    # Pour que la grille soit plus jolie
    pylab.scatter(thegrid[:,0],thegrid[:,1],c = classesPred, s=50)
    # Les points d'entrainment
    pylab.scatter(train[:,0], train[:,1], marker = 'd', s=50)
    # Les points de test
    pylab.scatter(test[:,0], test[:,1], marker = 'x', s=50)

    labels = ['grille','train','test']
    pylab.legend(labels)

    pylab.axis('equal')
    pylab.show()

## http://code.activestate.com/recipes/302478/
def combine(*seqin):
    '''returns a list of all combinations of argument sequences.
for example: combine((1,2),(3,4)) returns
[[1, 3], [1, 4], [2, 3], [2, 4]]'''
    def rloop(seqin,listout,comb):
        '''recursive looping function'''
        if seqin:                       # any more sequences to process?
            for item in seqin[0]:
                newcomb=comb+[item]     # add next item to current comb
                # call rloop w/ rem seqs, newcomb
                rloop(seqin[1:],listout,newcomb)
        else:                           # processing last sequence
            listout.append(comb)        # comb finished, add to list
    listout=[]                      # listout initialization
    rloop(seqin,listout,[])         # start recursive process
    return listout


# plot pour les courbes d'apprentissage (erreur et perte)
def learning_plot(x, train, valid, test, title):
    pylab.plot(x, train, label="Train")
    pylab.plot(x, valid, label="Validation")
    pylab.plot(x, test, label="Test")
    pylab.title(title)
    pylab.legend()

    pylab.show()
    
