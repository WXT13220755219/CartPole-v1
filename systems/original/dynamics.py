import numpy as np
from scipy.integrate import solve_ivp

class OriginalSystem:
    def __init__(self):
        pass # 无需特殊参数

    def dynamics(self, t, x, u):
        x1, x2 = x
        u = u.flatten() # 修复广播错误
        dx1 = x2 + u[0]
        dx2 = -0.5*x1 - x1**2 + u[1]
        return [dx1, dx2]

    def step(self, x_current, u_current, dt):
        u_val = np.array(u_current, dtype=np.float64)
        # 修复：使用更稳定的积分器
        sol = solve_ivp(self.dynamics, (0, dt), x_current, args=(u_val,), method='RK45', rtol=1e-6)
        return sol.y[:, -1]

    def get_jacobian(self, x, u):
        x1, x2 = x
        A_c = np.array([[0, 1], [-0.5 - 2*x1, 0]])
        B_c = np.array([[1, 0], [0, 1]])
        return A_c, B_c