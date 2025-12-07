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

    # ==========================================
    # 1. Franka 物理参数
    # ==========================================
    dt = 0.01          # 仿真步长 (机械臂需要较高的频率，建议 100Hz)
    sim_time = 5.0     # 评估时的仿真时长
    
    n_states = 14      # 7个关节角度 (q) + 7个关节速度 (dq)
    n_inputs = 7       # 7个关节力矩 (tau)
    
    # 关节限制 (参考 Franka Emika Panda 参数)
    q_min = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    q_max = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])
    torque_max = np.array([87, 87, 87, 87, 12, 12, 12]) # 简化的最大力矩限制

    # ==========================================
    # 2. SINDy 辨识参数
    # ==========================================
    # 注意：高维系统如果 poly_order 过高会导致特征爆炸
    # 14维状态下：
    # poly_order=1 + trig=True => 特征数约 15 (常数+线性) + 28 (三角) = 43 (推荐)
    # poly_order=2 + trig=True => 特征数约 120 (二次) + 28 (三角) = 148 (计算量较大但可接受)
    poly_order = 1     
    use_trig = True    # 机械臂动力学强依赖 sin/cos，必须开启
    
    threshold = 0.002  # 稀疏系数阈值
    lasso_alpha = 1e-3 # 正则化强度
    
    data_samples = 10000 # 采样点数 (高维空间需要更多数据)
    collect_range = 1.0  # 初始状态的随机范围 (弧度)

    # ==========================================
    # 3. MPC 控制参数
    # ==========================================
    mpc_horizon = 8    # 预测步长 (太长会显著增加求解耗时)
    Q_gain = 300.0       # 状态跟踪权重 (加大以保证精度)
    R_gain = 0.05         # 输入节能权重
    Rd_gain = 0.1        # 输入平滑权重
    Q_integral_gain = 100.0 # 积分误差权重