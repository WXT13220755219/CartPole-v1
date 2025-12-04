import numpy as np
from scipy.integrate import odeint

class OriginalSystem:
    """
    MIMO Nonlinear System (Hopf Bifurcation Normal Form)
    Source: SINDy-KMPC Repository
    """
    def __init__(self):
        # 系统参数 (来自 sindy-kmpc/dynamics.py)
        self.mu = 0.5
        self.omega = 1.0

    def dynamics(self, x, t, u):
        """ 非线性动力学方程 dx/dt = f(x, u) """
        x1, x2 = x
        u1, u2 = u
        
        # Hopf 分岔形式
        # dx1 = mu*x1 - omega*x2 - x1*(x1^2 + x2^2) + u1
        # dx2 = omega*x1 + mu*x2 - x2*(x1^2 + x2^2) + u2
        
        r2 = x1**2 + x2**2
        dx1 = self.mu * x1 - self.omega * x2 - x1 * r2 + u1
        dx2 = self.omega * x1 + self.mu * x2 - x2 * r2 + u2
        
        return [dx1, dx2]

    def step(self, x_current, u_current, dt):
        """ 仿真一步积分 """
        t_span = [0, dt]
        # 注意: odeint 的 args 需要是 tuple
        sol = odeint(self.dynamics, x_current, t_span, args=(u_current,))
        return sol[-1, :]

    def get_jacobian(self, x, u):
        """
        计算连续时间下的雅可比矩阵 A_c = df/dx, B_c = df/du
        用于 LTV-MPC 的线性化 (如果 Runner 中开启对比实验)
        """
        x1, x2 = x
        r2 = x1**2 + x2**2
        
        # df1/dx1 = mu - (3x1^2 + x2^2)
        J11 = self.mu - (3*x1**2 + x2**2)
        # df1/dx2 = -omega - 2x1x2
        J12 = -self.omega - 2*x1*x2
        
        # df2/dx1 = omega - 2x1x2
        J21 = self.omega - 2*x1*x2
        # df2/dx2 = mu - (x1^2 + 3x2^2)
        J22 = self.mu - (x1**2 + 3*x2**2)
        
        A_c = np.array([[J11, J12], [J21, J22]])
        B_c = np.eye(2) # u1, u2 直接加在 dx1, dx2 上
        
        return A_c, B_c
