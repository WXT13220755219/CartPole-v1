import os
import datetime
import numpy as np

class FrankaConfig:
    base_dir = "simulation_results"
    
    def __init__(self, exp_note=""):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{timestamp}_Franka_{exp_note}"
        self.results_dir = os.path.join(self.base_dir, folder_name)
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
    
    # --- Franka 机械臂参数 ---
    dt = 0.02           # 仿真控制步长 (20ms)
    sim_time = 4.0      # 总仿真时长 (不宜过长，计算量大)
    
    # 状态定义 (参考 Learn_Koopman_with_Klinear_Franka.py)
    # [EE_pos(3), Joint_pos(7), Joint_vel(7)] -> 共 17 维
    n_states = 17    
    n_inputs = 7     # 7个关节的力矩
    
    # 物理限制
    # 关节范围 (rad)
    joint_low = np.array([-2.9, -1.8, -2.9, -3.0, -2.9, -0.08, -2.9])
    joint_high = np.array([2.9, 1.8, 2.9, 0.08, 2.9, 3.0, 2.9])
    u_max = 50.0     # 力矩限制 (N·m)，根据具体任务调整
    
    # --- SINDy / Koopman 参数 ---
    # 17维数据的二阶多项式计算量较大。
    # 如果运行太慢，可尝试改 poly_order=1 (线性/DMD)
    poly_order = 2   
    use_trig = False # 机械臂建议关闭纯三角函数扩展，避免特征爆炸
    
    # --- MPC 参数 ---
    mpc_horizon = 5       # 预测步数 (由于矩阵大，步数不宜过多，否则求解器卡死)
    Q_gain = 50.0         # 状态误差惩罚
    R_gain = 0.01         # 控制量惩罚
    Rd_gain = 0.01        # 控制增量惩罚
    Q_integral_gain = 0.0 # 暂不使用积分项
    
    # --- 数据采集 ---
    threshold = 0.005     # SINDy 稀疏阈值
    data_samples = 6000   # 样本数量 (高维需要更多数据)
    lasso_alpha = 1e-5
    collect_range = 0.5   # 随机初始化的扰动范围