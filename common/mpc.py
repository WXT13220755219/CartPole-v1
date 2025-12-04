import numpy as np
import cvxpy as cp

class KoopmanMPC:
    """ 保持您原有的 KoopmanMPC 代码不变 """
    def __init__(self, config, koopman_model):
        self.N = config.mpc_horizon
        self.Q_gain = config.Q_gain
        self.R_gain = config.R_gain
        self.Rd_gain = config.Rd_gain
        self.Q_integral_gain = config.Q_integral_gain
        
        self.A = koopman_model.A
        self.B = koopman_model.B
        self.n_lifted = koopman_model.n_lifted
        self.n_inputs = config.n_inputs
        self.n_states = config.n_states
        
        self.C_lifted = np.hstack([np.eye(self.n_states), np.zeros((self.n_states, self.n_lifted - self.n_states))])

    def get_control(self, z_current, integral_error_prev, z_ref_horizon, x_ref_horizon, u_prev):
        # ... (您原有的 Koopman MPC 代码内容) ...
        # 为节省篇幅，此处省略，逻辑与之前一致
        z_var = cp.Variable((self.N + 1, self.n_lifted))
        epsilon_var = cp.Variable((self.N + 1, self.n_states))
        u_var = cp.Variable((self.N, self.n_inputs))
        
        cost = 0
        constraints = []
        
        constraints.append(z_var[0] == z_current)
        constraints.append(epsilon_var[0] == integral_error_prev)
        
        for k in range(self.N):
            constraints.append(z_var[k+1] == self.A @ z_var[k] + self.B @ u_var[k])
            x_k = self.C_lifted @ z_var[k]
            x_ref_k = x_ref_horizon[k]
            constraints.append(epsilon_var[k+1] == epsilon_var[k] + (x_k - x_ref_k))
            
            ref_z_k = z_ref_horizon[k]
            cost += self.Q_gain * cp.sum_squares(z_var[k] - ref_z_k)
            cost += self.Q_integral_gain * cp.sum_squares(epsilon_var[k])
            cost += self.R_gain * cp.sum_squares(u_var[k])
            
            if k == 0:
                delta_u = u_var[k] - u_prev
            else:
                delta_u = u_var[k] - u_var[k-1]
            cost += self.Rd_gain * cp.sum_squares(delta_u)
            
            constraints.append(u_var[k] <= 10.0) 
            constraints.append(u_var[k] >= -10.0)
            
        cost += self.Q_gain * cp.sum_squares(z_var[self.N] - z_ref_horizon[self.N])
        cost += self.Q_integral_gain * cp.sum_squares(epsilon_var[self.N])

        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.OSQP, warm_start=True)
        
        if prob.status != cp.OPTIMAL:
            return u_prev, integral_error_prev 
        
        return u_var[0].value, epsilon_var[1].value


class LinearMPC:
    """ 
    [新增] LTV-MPC (Linear Time-Varying MPC)
    作为对照实验，基于当前参考轨迹进行局部线性化
    """
    def __init__(self, config, system_model):
        self.N = config.mpc_horizon
        self.dt = config.dt
        self.Q_gain = config.Q_gain
        self.R_gain = config.R_gain
        self.Rd_gain = config.Rd_gain
        # LTV 同样可以加积分项来公平对比
        self.Q_integral_gain = config.Q_integral_gain 
        
        self.env = system_model
        self.n_inputs = config.n_inputs
        self.n_states = config.n_states

    def get_control(self, x_current, integral_error_prev, x_ref_horizon, u_prev):
        """
        x_current: (n_states,)
        x_ref_horizon: (N+1, n_states) 原始状态参考轨迹
        """
        # 变量定义
        x_var = cp.Variable((self.N + 1, self.n_states))
        epsilon_var = cp.Variable((self.N + 1, self.n_states)) # 积分误差
        u_var = cp.Variable((self.N, self.n_inputs))
        
        cost = 0
        constraints = []
        
        # 初始约束
        constraints.append(x_var[0] == x_current)
        constraints.append(epsilon_var[0] == integral_error_prev)
        
        for k in range(self.N):
            # === LTV 核心：在参考点处计算雅可比矩阵 ===
            x_ref_k = x_ref_horizon[k]
            # 假设参考输入为0 (或者可以传入 u_ref_horizon)
            u_ref_k = np.zeros(self.n_inputs) 
            
            # 1. 计算连续时间雅可比矩阵 A_c, B_c
            A_c, B_c = self.env.get_jacobian(x_ref_k, u_ref_k)
            
            # 2. 欧拉离散化: x(k+1) = (I + A_c*dt)*x(k) + (B_c*dt)*u(k) + d_k
            # 为了准确跟踪，采用偏差模型: 
            # x_{k+1} - x_ref_{k+1} = Ad * (x_k - x_ref_k) + Bd * (u_k - u_ref_k)
            # 展开后: x_{k+1} = Ad * x_k + Bd * u_k + [x_ref_{k+1} - Ad*x_ref_k - Bd*u_ref_k]
            
            Ad = np.eye(self.n_states) + A_c * self.dt
            Bd = B_c * self.dt
            
            # 仿射项 (Linearization Offset)
            # 这一项保证了如果系统处于参考轨迹上，误差为0
            affine_term = x_ref_horizon[k+1] - (Ad @ x_ref_k + Bd @ u_ref_k)
            
            # 动力学约束
            constraints.append(x_var[k+1] == Ad @ x_var[k] + Bd @ u_var[k] + affine_term)
            
            # 积分误差更新
            constraints.append(epsilon_var[k+1] == epsilon_var[k] + (x_var[k] - x_ref_k))
            
            # 代价函数 (与 Koopman 保持一致)
            cost += self.Q_gain * cp.sum_squares(x_var[k] - x_ref_k)
            cost += self.Q_integral_gain * cp.sum_squares(epsilon_var[k])
            cost += self.R_gain * cp.sum_squares(u_var[k])
            
            # 平滑项
            if k == 0:
                delta_u = u_var[k] - u_prev
            else:
                delta_u = u_var[k] - u_var[k-1]
            cost += self.Rd_gain * cp.sum_squares(delta_u)
            
            # 输入约束
            constraints.append(u_var[k] <= 10.0)
            constraints.append(u_var[k] >= -10.0)
            
        # 终端代价
        cost += self.Q_gain * cp.sum_squares(x_var[self.N] - x_ref_horizon[self.N])
        cost += self.Q_integral_gain * cp.sum_squares(epsilon_var[self.N])

        # 求解
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.OSQP, warm_start=True, max_iter=20000, eps_abs=1e-4, eps_rel=1e-4)
        
        if prob.status != cp.OPTIMAL:
            return u_prev, integral_error_prev
            
        return u_var[0].value, epsilon_var[1].value