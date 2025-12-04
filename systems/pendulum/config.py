import os
import datetime

class PendulumConfig:
    base_dir = "simulation_results"
    
    def __init__(self, exp_note=""):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{timestamp}_Pendulum_{exp_note}"
        self.results_dir = os.path.join(self.base_dir, folder_name)
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
    
    # --- 倒立摆专属参数 ---
    dt = 0.05
    sim_time = 10.0
    n_states = 2     # [theta, theta_dot]
    n_inputs = 1     # Torque
    
    # Koopman
    poly_order = 2   
    use_trig = True  # 开启三角函数
    
    # MPC
    mpc_horizon = 15
    Q_gain = 100.0        
    R_gain = 1.0          
    Rd_gain = 0.1         
    Q_integral_gain = 50.0 
    
    # SINDy
    threshold = 0.005 
    data_samples = 10000
    lasso_alpha = 1e-6
    collect_range = 3.14