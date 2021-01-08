import numpy as np
import activation_functions
import loss_functions
print(loss_functions.mean_square_loss(np.array([[0],[1],[2],[3],[4],[5],[6],[7],[8]], dtype = float),np.array([[0],[1],[2],[3],[4],[5],[6],[7],[6]], dtype = float),True))


