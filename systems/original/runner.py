import numpy as np
from common.koopman import SindyKoopman
from common.mpc import KoopmanMPC, LinearMPC
import common.utils as utils
from .config import OriginalConfig
from .dynamics import OriginalSystem

def run():
    print("=== Running Original System (MIMO Figure-8 Tracking) ===")
    cfg = OriginalConfig(exp_note="MIMO_Tracking")
    env = OriginalSystem()
    kp_model = SindyKoopman(cfg)

    # ==========================================
    # 1. 数据采集 (混合策略)
    # ==========================================
    # 复现 sindy-kmpc/main.py 的数据采集逻辑，这对于训练好的 SINDy 模型至关重要
    print("[Phase 1] Collecting training data & Training...")
    X_train, U_train, X_next_train = [], [], []
    x = np.array([0.1, 0.1])
    
    for i in range(cfg.data_samples):
        # 30% 概率使用纯随机激励，70% 概率使用正弦/余弦激励以覆盖频域特性
        if np.random.rand() > 0.7:
            u = np.random.uniform(-3, 3, size=(cfg.n_inputs,))
        else:
            freq1, freq2 = 0.5 + np.random.rand(), 0.2 + np.random.rand()
            u = np.array([
                2.0 * np.sin(freq1 * i * cfg.dt) + np.random.normal(0, 0.1),
                2.0 * np.cos(freq2 * i * cfg.dt) + np.random.normal(0, 0.1)
            ])
            
        x_next = env.step(x, u, cfg.dt)
        X_train.append(x); U_train.append(u); X_next_train.append(x_next)
        
        # 定期重置状态，防止发散过远
        if i % 200 == 0: 
            x = np.random.uniform(-1.5, 1.5, size=(cfg.n_states,))
        else: 
            x = x_next
            
    # 训练模型
    kp_model.fit(np.array(X_train), np.array(U_train), np.array(X_next_train))

    # ==========================================
    # 2. 可视化 Koopman 矩阵结构
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
    # 3. 闭环控制 (Figure-8 轨迹跟踪)
    # ==========================================
    print("[Phase 3] Control Evaluation (Figure-8 Tracking)...")
    
    # 生成参考轨迹 (8字形)
    # 这里的 horizon + 1 是为了保证 MPC 在最后几步也有参考值
    steps = int(cfg.sim_time / cfg.dt)
    t_eval = np.arange(0, cfg.sim_time + cfg.dt * cfg.mpc_horizon, cfg.dt)
    
    # 使用 common/utils.py 中的函数生成轨迹
    X_ref_full = utils.generate_figure_8_traj(t_eval, scale=1.5, omega=0.5)
    
    # 提升参考轨迹 (对于 Koopman MPC 是必须的)
    Z_ref_full = kp_model.lift(X_ref_full) 
    
    # --- Koopman MPC ---
    mpc_k = KoopmanMPC(cfg, kp_model)
    x_curr = X_ref_full[0].copy() # 从参考轨迹起点开始
    u_prev = np.zeros(cfg.n_inputs)
    int_err = np.zeros(cfg.n_states)
    
    log_x_k, log_u_k = [x_curr], []
    
    print("Running Koopman MPC loop...")
    for k in range(steps):
        z_curr = kp_model.lift(x_curr).flatten()
        
        # 获取 MPC 视界的参考片段
        z_ref_h = Z_ref_full[k : k + cfg.mpc_horizon + 1]
        x_ref_h = X_ref_full[k : k + cfg.mpc_horizon + 1]
        
        try:
            u_opt, int_err = mpc_k.get_control(z_curr, int_err, z_ref_h, x_ref_h, u_prev)
        except Exception as e:
            print(f"MPC Error at step {k}: {e}")
            u_opt = u_prev # 容错
            
        x_next = env.step(x_curr, u_opt, cfg.dt)
        log_x_k.append(x_next); log_u_k.append(u_opt)
        
        x_curr = x_next
        u_prev = u_opt

    # --- Linear MPC (作为 Benchmark，复现 sindy-kmpc 的对比) ---
    print("Running Linear MPC loop (Benchmark)...")
    mpc_l = LinearMPC(cfg, env)
    x_curr = X_ref_full[0].copy()
    u_prev = np.zeros(cfg.n_inputs)
    int_err = np.zeros(cfg.n_states)
    
    log_x_l, log_u_l = [x_curr], []
    
    for k in range(steps):
        x_ref_h = X_ref_full[k : k + cfg.mpc_horizon + 1]
        try:
            u_opt, int_err = mpc_l.get_control(x_curr, int_err, x_ref_h, u_prev)
        except:
            u_opt = u_prev
            
        x_next = env.step(x_curr, u_opt, cfg.dt)
        log_x_l.append(x_next); log_u_l.append(u_opt)
        
        x_curr = x_next
        u_prev = u_opt

    # ==========================================
    # 4. 绘图
    # ==========================================
    print("[Phase 4] Generating Plots...")
    min_len = min(len(log_x_k), len(log_x_l))
    u_len = min(len(log_u_k), len(log_u_l))
    t_plot = np.arange(0, cfg.sim_time + cfg.dt, cfg.dt)[:min_len]
    
    # 轨迹与控制对比
    utils.plot_trajectory_and_control(
        t_plot, X_ref_full[:min_len],
        np.array(log_u_k)[:u_len], np.array(log_x_k)[:min_len],
        np.array(log_u_l)[:u_len], np.array(log_x_l)[:min_len], 
        cfg.results_dir
    )
    
    # 误差指标对比
    utils.plot_error_and_metrics(
        t_plot, X_ref_full[:min_len],
        np.array(log_u_k)[:u_len], np.array(log_x_k)[:min_len],
        np.array(log_u_l)[:u_len], np.array(log_x_l)[:min_len], 
        cfg.results_dir
    )
    
    # 相平面图 (Figure 8 效果图)
    utils.plot_phase_plane_trajectory(
        X_ref_full[:min_len],          
        np.array(log_x_k)[:min_len],   
        np.array(log_x_l)[:min_len],   
        cfg.results_dir
    )
    
    print(f"Original System Simulation Done. Results saved in {cfg.results_dir}")
