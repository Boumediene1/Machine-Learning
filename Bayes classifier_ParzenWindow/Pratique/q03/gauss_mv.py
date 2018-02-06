import numpy as np

class gauss_mv:
    '''
    n_dims: dimensions de donnees
    variance (sigma_sq)
    mu: la moyenne
    sigma_sq: la variance
    '''
    def __init__(self,n_dims):
        self.n_dims = n_dims
        self.mu = np.zeros((1,n_dims))
        self.sigma_sq = np.ones(n_dims)
    
    def train(self, train_data):
        self.mu = np.mean(train_data,axis=0)
        # Calculer la variance de chaque dimension de donnees
        sigma_sq = np.var(train_data, axis=0)
        self.sigma_sq = sigma_sq
            
    '''
    Calculer le log de probabilite Guassienne Diagonale
	'''
    def compute_predictions(self, test_data):
        if self.n_dims == 1:
            c = -self.n_dims * np.log(2*np.pi)/2.0 - np.log(self.sigma_sq)/2.0
            log_prob = c - (test_data-self.mu)**2.0 / (2.0 * self.sigma_sq)

        else:
            c = -self.n_dims * np.log(2*np.pi)/2.0 - np.log(np.prod(self.sigma_sq))/2.0
            log_prob = c - np.sum((test_data -  self.mu)**2.0/ (2.0 * self.sigma_sq), axis=1)

        return log_prob
