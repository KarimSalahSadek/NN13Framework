import numpy as np
import activation_functions
import loss_functions
print(loss_functions.multinomial_loss(np.array([[13],[-11],[5],[-15],[-18]], dtype = float),np.array([[1],[1],[1],[1],[1]], dtype = float),True))


