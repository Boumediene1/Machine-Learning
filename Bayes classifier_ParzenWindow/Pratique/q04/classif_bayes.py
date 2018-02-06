import numpy as np

class classif_bayes:

    def __init__(self,modeles_mv, priors):
        self.modeles_mv = modeles_mv
        self.priors = priors
        if len(self.modeles_mv) != len(self.priors):
            print('Le nombre de modeles MV doit etre egale au nombre de priors!')
        self.n_classes = len(self.modeles_mv)
			
    
    def compute_predictions(self, test_data):
        # log_pred = np.empty((test_data.shape[0])

        log_pred = np.empty((test_data.shape[0],self.n_classes))

        for i in range(self.n_classes):
            log_pred[:,i] = self.modeles_mv[i].compute_predictions(test_data) +  np.log(self.priors[i])

        return log_pred
