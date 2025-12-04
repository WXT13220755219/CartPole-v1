import numpy as np
from common.koopman import SindyKoopman
from common.mpc import KoopmanMPC, LinearMPC
import common.utils as utils
from .config import OriginalConfig
from .dynamics import OriginalSystem

def run():
    print("=== Running Original System (Stabilization Task) ===")
    cfg = OriginalConfig(exp_note="stabilization")
    env = OriginalSystem()
    kp_model = SindyKoopman(cfg)

    # ==========================================
    # 1. 数据采集
    # ==========================================
    print("[Phase 1] Training...")
    X, U, Xn = [], [], []
    x = np.random.uniform(-cfg.collect_range, cfg.collect_range, size=(cfg.n_states,))
    
    for i in range(cfg.data_samples):
        u = np.random.uniform(-5, 5, size=(cfg.n_inputs,)) # 纯随机
        xn = env.step(x, u, cfg.dt)
        
        if np.any(np.abs(xn) > 1e3):
            x = np.random.uniform(-cfg.collect_range, cfg.collect_range, size=(cfg.n_states,))
            continue
            
        X.append(x); U.append(u); Xn.append(xn)
        
        if i % 50 == 0:
            x = np.random.uniform(-cfg.collect_range, cfg.collect_range, size=(cfg.n_states,))
        else:
            x = xn
            
    kp_model.fit(np.array(X), np.array(U), np.array(Xn))

    # ==========================================
    # [新增] 2. 可视化 Koopman 矩阵结构 (热力图)
    # ==========================================
    print("[Phase 2] Visualizing Structure (Heatmap)...")
    feat_names = kp_model.get_feature_names()
    inp_names = [f"u{i+1}" for i in range(cfg.n_inputs)]
    
    utils.plot_combined_structure_heatmap(
        kp_model.A, kp_model.B, 
        cfg.results_dir, 
        feat_names, 
        inp_names
    )

    # ==========================================
    # 3. 闭环控制
    # ==========================================
    print("[Phase 3] Control Evaluation...")
    steps = int(cfg.sim_time / cfg.dt)
    t_seq = np.arange(steps + cfg.mpc_horizon + 1) * cfg.dt
    
    # --- 构造零参考轨迹 ---
    X_ref = np.zeros((len(t_seq), cfg.n_states))
    Z_ref = kp_model.lift(X_ref)
    
    # --- Koopman MPC ---
    mpc_k = KoopmanMPC(cfg, kp_model)
    init_state = np.array([1.0, -0.5]) 
    x_curr = init_state.copy()
    u_prev = np.zeros(cfg.n_inputs)
    int_err = np.zeros(cfg.n_states)
    log_x_k, log_u_k = [x_curr], []

    for k in range(steps):
        z_curr = kp_model.lift(x_curr).flatten()
        try:
            u_opt, int_err = mpc_k.get_control(z_curr, int_err, Z_ref[k:k+cfg.mpc_horizon+1], X_ref[k:k+cfg.mpc_horizon+1], u_prev)
        except Exception as e:
            print(f"MPC Error: {e}"); break
            
        x_curr = env.step(x_curr, u_opt, cfg.dt)
        log_x_k.append(x_curr); log_u_k.append(u_opt)
        u_prev = u_opt

    # --- Linear MPC ---
    mpc_l = LinearMPC(cfg, env)
    x_curr = init_state.copy()
    u_prev = np.zeros(cfg.n_inputs)
    int_err = np.zeros(cfg.n_states)
    log_x_l, log_u_l = [x_curr], []

    for k in range(steps):
        try:
            u_opt, int_err = mpc_l.get_control(x_curr, int_err, X_ref[k:k+cfg.mpc_horizon+1], u_prev)
        except: break
        x_curr = env.step(x_curr, u_opt, cfg.dt)
        log_x_l.append(x_curr); log_u_l.append(u_opt)
        u_prev = u_opt

    # 绘图
    t_plot = np.arange(0, cfg.sim_time + cfg.dt, cfg.dt)[:len(log_x_k)]
    utils.plot_trajectory_and_control(t_plot, X_ref[:len(log_x_k)], np.array(log_u_k), np.array(log_x_k), np.array(log_u_l), np.array(log_x_l), cfg.results_dir)
    print("Original System Done.")