import autograd.numpy as np
from activation_functions import *

iterations = 1500
learning_rate = 0.01
cost_f = Beale()
# from act. fn(fr or bk)

class SGD(Optimizer):

    def __init__(self, cost_f, lr=0.001, x=None, y=None):
        super().__init__(cost_f, lr, x, y)



def step(self, lr = None):
    if not lr:
        lr = self.lr # eita
    f = cost_f.eval(self.x, self.y)
    dx = cost_f.df_dx(self.x, self.y)
    dy = cost_f.df_dy(self.x, self.y)
    self.x = self.x - lr *dx
    self.y = self.y - lr *dy
    return [self.x, self.y]
opt = SGD(cost_f=cost_f, lr=learning_rate)


class SGD_momentum(Optimizer):
    def __init__(self, cost_f, lr=0.001, beta=0.9, x=None, y=None):
        super().__init__(cost_f=cost_f, lr=lr, x=x, y=y, beta=beta)
        self.vx = 0
        self.vy = 0

    def step(self, lr=None, beta=None):
        if type(lr) == type(None):
            lr = self.lr
        if type(beta) == type(None):
            beta = self.beta
        f = self.cost_f.eval(self.x, self.y)
        dx = self.cost_f.df_dx(self.x, self.y)
        dy = self.cost_f.df_dy(self.x, self.y)

        self.vx = beta * self.vx + lr * dx
        self.vy = beta * self.vy + lr * dy
        self.x += - self.vx
        self.y += - self.vy

        return [self.x, self.y]

    opt = SGD_momentum(cost_f=cost_f, lr=learning_rate, beta=0.9)


class SGD_nesterov_momentum(Optimizer):
    def __init__(self, cost_f, lr=0.001, beta=0.9, x=None, y=None):
        super().__init__(cost_f=cost_f, lr=lr, x=x, y=y, beta=beta)
        self.vx = None

    def step(self, lr=None, beta=None):
        f = self.cost_f.eval(self.x, self.y)
        dx = self.cost_f.df_dx(self.x, self.y)
        dy = self.cost_f.df_dy(self.x, self.y)

        if type(lr) == type(None):
            lr = self.lr
        if type(beta) == type(None):
            beta = self.beta
        if type(self.vx) == type(None) or type(self.vy) == type(None):
            self.vx = lr * dx
            self.vy = lr * dy
        else:
            dx_in_vx = self.cost_f.df_dx(self.x - beta * self.vx, self.y - beta * self.vy)
            dy_in_vy = self.cost_f.df_dy(self.x - beta * self.vx, self.y - beta * self.vy)
            self.vx = beta * self.vx + lr * dx_in_vx
            self.vy = beta * self.vy + lr * dy_in_vy
        self.x += - self.vx
        self.y += - self.vy

        return [self.x, self.y]
opt = SGD_nesterov_momentum(cost_f=cost_f, lr=learning_rate, beta=0.9)


class AdaGrad(Optimizer):
    def __init__(self, cost_f, lr=0.001, x=None, y=None):
        super().__init__(cost_f=cost_f, lr=lr, x=x, y=y)
        self.sumsq_dx = 0
        self.sumsq_dy = 0

    def step(self, lr=None):
        epsilon = 1e-8
        if not lr:
            lr = self.lr
        # derivative
        f = self.cost_f.eval(self.x, self.y)
        dx = self.cost_f.df_dx(self.x, self.y)
        dy = self.cost_f.df_dy(self.x, self.y)
        self.sumsq_dx += dx ** 2
        self.sumsq_dy += dy ** 2
        self.x = self.x - (lr / (np.sqrt(epsilon + self.sumsq_dx))) * dx
        self.y = self.y - (lr / (np.sqrt(epsilon + self.sumsq_dy))) * dy

        return [self.x, self.y]


opt = AdaGrad(cost_f=cost_f, lr=learning_rate)



class RMSProp(Optimizer):
    def __init__(self, cost_f, lr=0.001, decay_rate=0.9, x=None, y=None):
        super().__init__(cost_f=cost_f, lr=lr, x=x, y=y, decay_rate=decay_rate)
        self.ms_x = 0
        self.ms_y = 0

    def step(self, lr=None, decay_rate=None):
        epsilon = 1e-8
        if not lr:
            lr = self.lr
        if not decay_rate:
            decay_rate = self.decay_rate
        # derivative
        f = self.cost_f.eval(self.x, self.y)
        dx = self.cost_f.df_dx(self.x, self.y)
        dy = self.cost_f.df_dy(self.x, self.y)
        self.ms_x = self.decay_rate * (self.ms_x) + (1 - self.decay_rate) * dx ** 2
        self.ms_y = self.decay_rate * (self.ms_y) + (1 - self.decay_rate) * dy ** 2
        self.x = self.x - (lr / (epsilon + np.sqrt(self.ms_x))) * dx
        self.y = self.y - (lr / (epsilon + np.sqrt(self.ms_y))) * dy

        return [self.x, self.y]

    opt = RMSProp(cost_f=cost_f, lr=learning_rate)



class AdaDelta(Optimizer):
    def __init__(self, cost_f, lr=0.001, decay_rate=0.9, x=None, y=None):
        super().__init__(cost_f=cost_f, lr=lr, x=x, y=y, decay_rate=decay_rate)
        self.decay_x = 0
        self.decay_y = 0
        self.decay_dx = 1
        self.decay_dy = 1

    def step(self, lr=None, decay_rate=None):
        epsilon = 1e-8
        if not lr:
            lr = self.lr
        if not decay_rate:
            decay_rate = self.decay_rate
        # derivative
        f = self.cost_f.eval(self.x, self.y)
        dx = self.cost_f.df_dx(self.x, self.y)
        dy = self.cost_f.df_dy(self.x, self.y)
        # Update decays
        self.decay_x = decay_rate * (self.decay_x) + (1 - decay_rate) * dx ** 2
        self.decay_y = decay_rate * (self.decay_y) + (1 - decay_rate) * dy ** 2

        update_x = dx * ((np.sqrt(epsilon + self.decay_dx)) / (np.sqrt(epsilon + self.decay_x)))
        update_y = dy * ((np.sqrt(epsilon + self.decay_dy)) / (np.sqrt(epsilon + self.decay_y)))

        self.x = self.x - (update_x) * lr
        self.y = self.y - (update_y) * lr

        # Update decays d
        self.decay_dx = decay_rate * (self.decay_dx) + (1 - decay_rate) * update_x ** 2
        self.decay_dy = decay_rate * (self.decay_dy) + (1 - decay_rate) * update_y ** 2

        return [self.x, self.y]

    learning_rate = 0.01

    opt = AdaDelta(cost_f=cost_f, lr=learning_rate)




b1 = 0.9; b2 = 0.999; t = np.arange(1,6000)
bias_correction = np.sqrt(1-b2**t)/(1-b1**t)


class Adam(Optimizer):
    def __init__(self, cost_f, lr=0.001, beta_1=0.9, beta_2=0.999, x=None, y=None):
        super().__init__(cost_f, lr, x, y, beta_1=beta_1, beta_2=beta_2)
        self.m_x, self.m_y, self.v_x, self.v_y, self.t = 0, 0, 0, 0, 0

    def step(self, lr=None):
        self.t += 1
        epsilon = 1e-8
        if not lr:
            lr = self.lr
        # derivative
        f = self.cost_f.eval(self.x, self.y)
        dx = self.cost_f.df_dx(self.x, self.y)
        dy = self.cost_f.df_dy(self.x, self.y)

        self.m_x = self.beta_1 * self.m_x + (1 - self.beta_1) * dx
        self.m_y = self.beta_1 * self.m_y + (1 - self.beta_1) * dy
        self.v_x = self.beta_2 * self.v_x + (1 - self.beta_2) * (dx ** 2)
        self.v_y = self.beta_2 * self.v_y + (1 - self.beta_2) * (dy ** 2)

        m_x_hat = self.m_x / (1 - self.beta_1 ** self.t)
        m_y_hat = self.m_y / (1 - self.beta_1 ** self.t)
        v_x_hat = self.v_x / (1 - self.beta_2 ** self.t)
        v_y_hat = self.v_y / (1 - self.beta_2 ** self.t)

        self.x = self.x - (lr * m_x_hat) / (np.sqrt(v_x_hat) + epsilon)
        self.y = self.y - (lr * m_y_hat) / (np.sqrt(v_y_hat) + epsilon)
        return [self.x, self.y]

    opt = Adam(cost_f=cost_f, lr=learning_rate)




class AdaMax(Optimizer):
    def __init__(self, cost_f, lr=0.001, beta_1=0.9, beta_2=0.999, x=None, y=None):
        super().__init__(cost_f, lr, x, y, beta_1=beta_1, beta_2=beta_2)
        self.m_x, self.m_y, self.v_x, self.v_y, self.t = 0, 0, 0, 0, 0

    def step(self, lr=None):
        self.t += 1
        epsilon = 1e-8
        if not lr:
            lr = self.lr
        # derivative
        f = self.cost_f.eval(self.x, self.y)
        dx = self.cost_f.df_dx(self.x, self.y)
        dy = self.cost_f.df_dy(self.x, self.y)

        self.m_x = self.beta_1 * self.m_x + (1 - self.beta_1) * dx
        self.m_y = self.beta_1 * self.m_y + (1 - self.beta_1) * dy
        self.v_x = max(self.beta_2 * self.v_x, abs(dx))
        self.v_y = max(self.beta_2 * self.v_y, abs(dy))

        m_x_hat = self.m_x / (1 - self.beta_1 ** self.t)
        m_y_hat = self.m_y / (1 - self.beta_1 ** self.t)

        self.x = self.x - (lr * m_x_hat) / (self.v_x + epsilon)
        self.y = self.y - (lr * m_y_hat) / (self.v_y + epsilon)
        return [self.x, self.y]

    opt = AdaMax(cost_f=cost_f, lr=learning_rate)



class NAdam(Optimizer):
    def __init__(self, cost_f, lr=0.001, beta_1=0.9, beta_2=0.999, x=None, y=None):
        super().__init__(cost_f, lr, x, y, beta_1=beta_1, beta_2=beta_2)
        self.m_x, self.m_y, self.v_x, self.v_y, self.t = 0, 0, 0, 0, 0

    def step(self, lr=None):
        self.t += 1
        epsilon = 1e-8
        if not lr:
            lr = self.lr
        # derivative
        f = self.cost_f.eval(self.x, self.y)
        dx = self.cost_f.df_dx(self.x, self.y)
        dy = self.cost_f.df_dy(self.x, self.y)

        dx_hat = dx / (1 - self.beta_1 ** self.t)
        dy_hat = dy / (1 - self.beta_1 ** self.t)

        self.m_x = self.beta_1 * self.m_x + (1 - self.beta_1) * dx
        self.m_y = self.beta_1 * self.m_y + (1 - self.beta_1) * dy
        self.v_x = self.beta_2 * self.v_x + (1 - self.beta_2) * (dx ** 2)
        self.v_y = self.beta_2 * self.v_y + (1 - self.beta_2) * (dy ** 2)

        m_x_hat = self.m_x / (1 - self.beta_1 ** self.t)
        m_y_hat = self.m_y / (1 - self.beta_1 ** self.t)
        v_x_hat = self.v_x / (1 - self.beta_2 ** self.t)
        v_y_hat = self.v_y / (1 - self.beta_2 ** self.t)

        m_x_dash = (1 - self.beta_1) * dx_hat + self.beta_1 * m_x_hat
        m_y_dash = (1 - self.beta_1) * dy_hat + self.beta_1 * m_y_hat

        self.x = self.x - (lr * m_x_dash) / (np.sqrt(v_x_hat) + epsilon)
        self.y = self.y - (lr * m_y_dash) / (np.sqrt(v_y_hat) + epsilon)
        return [self.x, self.y]
opt = NAdam(cost_f=cost_f, lr=learning_rate)


