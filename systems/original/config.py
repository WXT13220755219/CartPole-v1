import os
import datetime

class OriginalConfig:
    base_dir = "simulation_results"
    
    def __init__(self, exp_note=""):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{timestamp}_Original_{exp_note}"
        self.results_dir = os.path.join(self.base_dir, folder_name)
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    # --- 原始系统专属参数 ---
    dt = 0.05
    sim_time = 20.0
    n_states = 2
    n_inputs = 2  
    
    # Koopman
    poly_order = 2 
    use_trig = False # 不需要三角函数
    
    # MPC
    mpc_horizon = 15
    Q_gain = 21.0
    R_gain = 0.6
    Rd_gain = 0.1
    Q_integral_gain = 10.0
    
    # SINDy
    threshold = 0.005
    data_samples = 6000
    lasso_alpha = 1e-5
    collect_range = 1.5