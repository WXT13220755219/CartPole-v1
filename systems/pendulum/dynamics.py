import numpy as np

class PendulumSystem:
    def __init__(self):
        self.g = 10.0
        self.m = 1.0
        self.l = 1.0
        self.max_torque = 8.0
        self.max_speed = 8.0

    def step(self, x_current, u_current, dt):
        th, thdot = x_current
        u = np.clip(u_current[0], -self.max_torque, self.max_torque)
        g, m, l = self.g, self.m, self.l
        
        newthdot = thdot + (3 * g / (2 * l) * np.sin(th) + 3.0 / (m * l ** 2) * u) * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * dt
        
        return np.array([newth, newthdot])

    def get_jacobian(self, x, u):
        # Linear MPC 线性化模型
        x1, x2 = x
        coeff_sin = 3 * self.g / (2 * self.l)
        coeff_u = 3.0 / (self.m * self.l ** 2)
        
        A_c = np.array([[0, 1], [coeff_sin * np.cos(x1), 0]])
        B_c = np.array([[0], [coeff_u]])
        return A_c, B_c