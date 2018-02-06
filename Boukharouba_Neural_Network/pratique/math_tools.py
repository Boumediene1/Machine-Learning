###################################################
# IFT3395 - devoir 3
# Auteurs: Boumedienne Boukharouba, Farzin Faridfar
# 4 decembre 2017
###################################################
import numpy as np

def softmax(data):
    sum_result = np.sum(np.exp(data), axis=0)
    result = (np.exp(data))/sum_result
    return result

def rect(data):
    result = data * (data > 0)
    return result

def linearize(data, weights, bias):
    result = np.dot(weights, data) + bias
    return result

def sign(x):
    return np.where(x > 0 , 1, -1)

def onehot(y, m):
    # Si y est un integer, creer un array a partir de lui
    if not isinstance(y, np.ndarray):
        y = np.array([y])
    
    onehot = np.zeros((y.shape[0], m))
    
    for i,row in enumerate(onehot):
        row[y[i]] = 1
    
    return onehot.T

# Cette fonction cree une matrice qui contient les lots
# de taille k*d
def creat_batch(data, k, d):
    # nombre total des exemples
    n_sample = data.shape[0]
    # nombre d'exemple dans chaque batch
    l = n_sample // k

    # matrice des bataches
    batch_mat = np.ones((k, l, d))

    # mettre les k exemple de donne dans chacun des lots
    for batch in range(k):
        batch_mat[batch] = np.array(data[batch*l:(batch*l+l),:])
    
    return batch_mat

# Cette fonction calcul les gradient selon la  methode de difference finie
def finite_diff(NN, finite_data, k, epsilon=10**-5):
    # creer la matrice des lots
    batch_mat = creat_batch(finite_data, k, finite_data.shape[1])

    for batch in batch_mat:
        batch_size = batch_mat[0].shape[0]

        # separer les traits et les cibles de donnees
        x_set = batch_mat[0][:, :-1]
        y_set = (batch_mat[0][:, -1]).astype(int)
        
        # Calculer le fprop avec les params actuels
        NN.fprop(x_set)
        loss_1 = -np.log(NN.os)

        # Calculer le gradient par back-propagation
        NN.bprop(x_set, y_set)
        # List des gradients calcule avec les params actuels
        grad_list = [NN.grad_w1, NN.grad_b1,
                         NN.grad_w2, NN.grad_b2]

        # Lest des parametres pour lesquels les gradient
        # seront calcules
        param_list = [NN.w1, NN.b1, NN.w2, NN.b2]

        # la matrice qui contient le ratio de chacun des gradients
        # calcule par la methode retropropagation par rapport a la
        # methode de la difference finie
        param_ratio = []
        for grad,param in enumerate(param_list):
            shape = param.shape
            # initialiser la matrice de chaque parametre
            finite_mat = np.zeros(shape)
            # pour chaque element scalaire dans la matrice
            # de parametre, on calcule la difference finie
            for i in range(shape[0]):
                for j in range(shape[1]):
                    # Ajouter epsilon aux parametres
                    param[i][j] += epsilon
                    NN.fprop(x_set)
                    param[i][j] -= epsilon
                    loss_2 = -np.log(NN.os)
                                                
                    finite_mat[i][j] = (loss_2[y_set][0]
                                            - loss_1[y_set][0])/epsilon
                    
                    ratio = finite_mat / grad_list[grad]

                    # si les deux resultat sont 0, numpy produit
                    # nan, dans cette ligne on les remplace par 1
                    gradFinite_ratio = np.where(
                        np.isnan(ratio),1,ratio)
                    
            param_ratio.append(gradFinite_ratio)
            
        return param_ratio

# Cette method affiche les dimension de tous les parametres
# (pour repondre a la question 6 de la partie pratique)
def printDimensions(NN, x_dim=(1,2)):
    print("x", str(x_dim).rjust(31))
    print("dh", str(NN.dh).rjust(26))
    print("m", str(NN.m).rjust(27))
    print("w1", str(NN.w1.shape).rjust(30))
    print("b1", str(NN.b1.shape).rjust(30))
    print("ha = w1.x+b1", str(NN.ha.shape).rjust(20))
    print("hs = rect(ha)", str(NN.hs.shape).rjust(19))
    print("w2", str(NN.w2.shape).rjust(30))
    print("b2", str(NN.b2.shape).rjust(30))
    print("oa = w2.hs+b2", str(NN.oa.shape).rjust(19))
    print("os = softmax(oa)", str(NN.os.shape).rjust(16))
    print("grad_oa = grad_os-onehot", str(NN.grad_oa.shape).rjust(8))
    print("grad_w2 = grad_oa.hs_T", str(NN.grad_w2.shape).rjust(10))
    print("grad_b2 = grad_oa", str(NN.grad_b2.shape).rjust(15))
    print("grad_hs = w2_T.grad_oa", str(NN.grad_hs.shape).rjust(10))
    print("grad_ha = 1_ha.grad_hs", str(NN.grad_ha.shape).rjust(10))
    print("grad_w1 = grad_ha.x_T", str(NN.grad_w1.shape).rjust(11))
    print("grad_b1 = grad_ha", str(NN.grad_b1.shape).rjust(15))


# Pour debauger
def printParams(NN):
    print("w1", str(NN.w1).rjust(30))
    print("b1", str(NN.b1).rjust(30))
    print("w2", str(NN.w2).rjust(30))
    print("b2", str(NN.b2).rjust(30))

# Pour debauger
def printAll(NN, x=np.ones((1,2))):
    print("x\n", x.T)
    print("dh\n",NN.dh)
    print("m\n", NN.m)
    print("w1\n", NN.w1)
    print("b1\n", NN.b1)
    print("ha = w1.x\n", NN.ha)
    print("hs = rect(ha)\n", NN.hs)
    print("w2\n", NN.w2)
    print("b2\n", NN.b2)
    print("oa = w2.hs\n", NN.oa)
    print("os = softmax(oa)\n", NN.os)
    print("grad_oa = grad_os-onehot\n", NN.grad_oa)
    print("grad_w2 = grad_oa.hs_T\n", NN.grad_w2)
    print("grad_b2 = grad_oa\n", NN.grad_b2)
    print("grad_hs = w2_T.grad_oa\n", NN.grad_hs)
    print("grad_ha = 1_ha.grad_hs\n", NN.grad_ha)
    print("grad_w1 = grad_ha.x_T\n", NN.grad_w1)
    print("grad_b1 = grad_ha\n", NN.grad_b1)
