import numpy as np
from numpy import linalg as LA

class parzen_mv:

    def __init__(self, n_dims, h):
        self.n_dims = n_dims
        self.mu = np.zeros((1,n_dims))
        self.h = h
            
    def train(self, train_data):
        self.mu = np.mean(train_data,axis=0)
        self.train_data = train_data
                    
    '''
    Calculer le log de probabilite Guassienne
	'''
    def compute_predictions(self, test_data):
        n_train = self.train_data.shape[0]
        n_test = test_data.shape[0]
        
        log_prob = np.zeros(test_data.shape[0])
        c = -np.log(n_test * (2*np.pi)**(self.n_dims/2.0)*self.h**self.n_dims)
        for i,test in enumerate(test_data):
            dist_sum = 0
            for train in self.train_data:
                dist_sum += np.exp(-(LA.norm(test-train))**2 / (2.0 * self.h**2))
                if dist_sum <= 0:
                    log_prob[i] = -np.inf
                else:
                    log_prob[i] = c + np.log(dist_sum)

        return log_prob
