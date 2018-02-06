import numpy as np
from gauss_mv import *
from parzen_mv import *

class model:
    def __init__(self, train_data, class_list):
        self.n_class = len(class_list)
        
        priors = []
        ordered_data = []
        for c in class_list:
            ordered_data.append([])
            priors.append(1)

        for ex in train_data:
            c = ex[-1]
            ex = ex[:-1]
            ordered_data[int(c-1)].append(ex.tolist())

        for i in range(len(class_list)):
            priors[i] = len(ordered_data[i])/train_data.shape[0]
            ordered_data[i] = np.array(ordered_data[i])

        self.ordered_data = ordered_data
        self.priors = priors

    def create_kernel(self, kernel, args, cols):
        self.kernel = kernel
        self.kernel_data_list = []
        dimension = len(cols)
        m = None
        if self.kernel == "gaussien":
            for i in range(self.n_class):
                m = gauss_mv(args[0])
                m.train((self.ordered_data[i])[:,cols])
                self.kernel_data_list.append(m)
                
        elif self.kernel == "parzen":
            for i in range(self.n_class):
                m = parzen_mv(args[0], args[1])
                m.train((self.ordered_data[i])[:,cols])
                self.kernel_data_list.append(m)
        
