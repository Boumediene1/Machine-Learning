import numpy as np
import matplotlib.pyplot as plt

from gauss_mv import *
from parzen_mv import *
from classif_bayes import *
from model import *
import utilitaires

# Ouvrir le fichier iris.txt
iris=np.loadtxt('iris.txt')

# Melanger les exemple d'Iris
np.random.seed(123)
indices = np.arange(0,150)
np.random.shuffle(indices)

# Les ensemble d'entrainement et de validation
iris_train = iris[indices[0:75]]
iris_valid = iris[indices[75:150]]

def class_predict(classifieur, data, cols):
    log_prob = classifieur.compute_predictions(data[:,cols])
    classesPred = log_prob.argmax(axis=1)+1
    return classesPred

def err_rate_print(classesPred, ref_class, phase):
    err_rate = (1-(classesPred==ref_class).mean())*100.0
    print ("Taux d'erreur ({0}) {1:.2f}".format(phase, err_rate))
    
def err_rate(model, sigma_list, data, cols):
    err_list = []
    for sigma in sigma_list:
        model.create_kernel("parzen", [2, sigma], cols)
        classifieur = classif_bayes(model.kernel_data_list, model.priors)
        classesPred = class_predict(classifieur, data, cols)
        err = (1-(classesPred==data[:,-1]).mean())*100.0
        err_list.append(err)
    return err_list

model = model(iris_train, [1,2,3])

cols_cmp = [0,1,2,3]
cols_2d = [0,1]

########################## NOYAU GAUSSIEN ##########################
## Complet
model.create_kernel("gaussien", [4],cols_cmp)
classifieur = classif_bayes(model.kernel_data_list, model.priors)
classesPred_train = class_predict(classifieur, iris_train, cols_cmp)
classesPred_valid = class_predict(classifieur, iris_valid, cols_cmp)

print("Bayes avec le noyeau Gaussien - Complet:")
err_rate_print(classesPred_train, iris_train[:,-1], "entrainement")
err_rate_print(classesPred_valid, iris_valid[:,-1], "validation")

print("----------------------------------")

## 2 Dimensions
model.create_kernel("gaussien", [2],cols_2d)
classifieur_g_2d = classif_bayes(model.kernel_data_list, model.priors)
classesPred_train = class_predict(classifieur_g_2d, iris_train, cols_2d)
classesPred_valid = class_predict(classifieur_g_2d, iris_valid, cols_2d)
print("Bayes avec le noyeau Gaussien - 2D:")
err_rate_print(classesPred_train, iris_train[:,-1], "entrainement")
err_rate_print(classesPred_valid, iris_valid[:,-1], "validation")

print("==================================")

########################## NOYAU PARZEN ##########################
## Complet
model.create_kernel("parzen", [2, 0.1], cols_cmp)
classifieur = classif_bayes(model.kernel_data_list, model.priors)
classesPred_train = class_predict(classifieur, iris_train, cols_cmp)
classesPred_valid = class_predict(classifieur, iris_valid, cols_cmp)
print("Bayes avec le noyeau Parzen - Complet:")
err_rate_print(classesPred_train, iris_train[:,-1], "entrainement")
err_rate_print(classesPred_valid, iris_valid[:,-1], "validation")

print("----------------------------------")

## 2 Dimensions
model.create_kernel("parzen", [2, 0.1], cols_2d)
classifieur_p_2d = classif_bayes(model.kernel_data_list, model.priors)
classesPred_train = class_predict(classifieur_p_2d, iris_train, cols_2d)
classesPred_valid = class_predict(classifieur_p_2d, iris_valid, cols_2d)
print("Bayes avec le noyeau Parzen - 2D:")
err_rate_print(classesPred_train, iris_train[:,-1], "entrainement")
err_rate_print(classesPred_valid, iris_valid[:,-1], "validation")

########################## TAUX D'ERREUR ##########################
sigma_list = np.linspace(0.05,0.5,100)

err_train = []
err_valid = []
## 2 Dimensions
err_train_2d = err_rate(model, sigma_list, iris_train, cols_2d)
err_valid_2d = err_rate(model, sigma_list, iris_valid, cols_2d)
## Complet
err_train_cmp = err_rate(model, sigma_list, iris_train, cols_cmp)
err_valid_cmp = err_rate(model, sigma_list, iris_valid, cols_cmp)

f, axarr = plt.subplots(2, sharex=True, figsize=(7, 7))

## Le courbe du taux d'erreur pour le 2 Diemensions
axarr[0].plot(sigma_list, err_train_2d, '-.g', label="Entrainement")
axarr[0].plot(sigma_list, err_valid_2d, '--y', label="Validation")
axarr[0].legend()
axarr[0].set_title('2D')
axarr[0].set_ylabel("Taux d'erreur")

## Le courbe du taux d'erreur pour le Complet
axarr[1].plot(sigma_list, err_train_cmp, '-.g', label="Entrainement")
axarr[1].plot(sigma_list, err_valid_cmp, '--y', label="Validation")
axarr[1].legend()
axarr[1].set_title('Complet')
axarr[1].set_xlabel('Sigma')
axarr[1].set_ylabel("Taux d'erreur")

plt.show()

## Affichage des regions de decisons
utilitaires.gridplot([classifieur_g_2d, classifieur_p_2d],
                     iris_train[:, cols_2d + [-1]],
                     iris_valid[:, cols_2d + [-1]],
                     n_points=50)

####################################################################
'''
Réponse de la question 4)
Le meilleur choix serait le classifieur Bayes avec le noyau Parzen
dans le cas ou on a les entrées complètes (les 4 traits).
Il ne faut pas oublier l’effet d’un sigma approprié pour
éviter le sur-apprentissage (s’il est très petit) et
le sous-apprentissage (s’il est très grand).

Bayes avec le noyeau Parzen - Complet:
Taux d'erreur (entrainement) 0.00
Taux d'erreur (validation) 5.33
'''
