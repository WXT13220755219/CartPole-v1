import numpy as np
import matplotlib.pyplot as plt
import os
from common.koopman import SindyKoopman
from common.mpc import KoopmanMPC, LinearMPC
import common.utils as utils
from .config import PendulumConfig
from .dynamics import PendulumSystem

def run():
    print("=== Running Pendulum System (Tracking Task) ===")
    cfg = PendulumConfig(exp_note="tracking")
    env = PendulumSystem()
    kp_model = SindyKoopman(cfg)

    # ==========================================
    # 1. 数据采集 (混合策略)
    # ==========================================
    print("[Phase 1] Training...")
    X, U, Xn = [], [], []
    x = np.random.uniform(-cfg.collect_range, cfg.collect_range, size=(cfg.n_states,))
    
    for i in range(cfg.data_samples):
        # 混合激励：随机输入 + 正弦输入，激发系统的非线性特性
        if np.random.rand() > 0.5:
            u = np.random.uniform(-8, 8, size=(cfg.n_inputs,))
        else:
            u = np.zeros(cfg.n_inputs)
            # 正弦激励
            u[0] = 5.0 * np.sin(0.5 * i * cfg.dt) + np.random.normal(0, 0.1)
        
        xn = env.step(x, u, cfg.dt)
        X.append(x); U.append(u); Xn.append(xn)
        
        # 定期重置或防发散
        if i % 50 == 0 or np.abs(xn[0]) > 4.0:
            x = np.random.uniform(-cfg.collect_range, cfg.collect_range, size=(cfg.n_states,))
        else:
            x = xn
    
    kp_model.fit(np.array(X), np.array(U), np.array(Xn))

    # ==========================================
    # 2. 可视化 Koopman 矩阵结构
    # ==========================================
    print("[Phase 2] Visualizing Structure (Heatmaps)...")
    feat_names = kp_model.get_feature_names()
    inp_names = [f"u{i+1}" for i in range(cfg.n_inputs)]
    
    # [新增] 分离视图：更清晰地查看 A 和 B 矩阵
    utils.plot_separated_AB_heatmaps(
        kp_model.A, kp_model.B, 
        cfg.results_dir, 
        feat_names, inp_names
    )
    
    # 全景视图
    utils.plot_combined_structure_heatmap(
        kp_model.A, kp_model.B, 
        cfg.results_dir, 
        feat_names, inp_names
    )

    # ==========================================
    # 3. 闭环控制 (正弦跟踪)
    # ==========================================
    print("[Phase 3] Control Evaluation...")
    steps = int(cfg.sim_time / cfg.dt)
    t_seq = np.arange(steps + cfg.mpc_horizon + 1) * cfg.dt
    
    # --- 构造正弦参考轨迹 (Phase Space 中的椭圆) ---
    A_ref, omega_ref = 1.5, 1.0
    ref_theta = A_ref * np.sin(omega_ref * t_seq)
    ref_theta_dot = A_ref * omega_ref * np.cos(omega_ref * t_seq)
    X_ref = np.vstack([ref_theta, ref_theta_dot]).T
    Z_ref = kp_model.lift(X_ref)
    
    # --- Koopman MPC ---
    print("Running Koopman MPC...")
    mpc_k = KoopmanMPC(cfg, kp_model)
    x_curr = X_ref[0].copy()
    u_prev = np.zeros(cfg.n_inputs)
    int_err = np.zeros(cfg.n_states)
    log_x_k, log_u_k = [x_curr], []

    for k in range(steps):
        z_curr = kp_model.lift(x_curr).flatten()
        try:
            u_opt, int_err = mpc_k.get_control(z_curr, int_err, Z_ref[k:k+cfg.mpc_horizon+1], X_ref[k:k+cfg.mpc_horizon+1], u_prev)
        except Exception as e:
            print(f"Koopman MPC Warning at step {k}: {e}")
            u_opt = u_prev
            
        x_curr = env.step(x_curr, u_opt, cfg.dt)
        log_x_k.append(x_curr); log_u_k.append(u_opt)
        u_prev = u_opt

    # --- Linear MPC ---
    print("Running Linear MPC...")
    mpc_l = LinearMPC(cfg, env)
    x_curr = X_ref[0].copy()
    u_prev = np.zeros(cfg.n_inputs)
    int_err = np.zeros(cfg.n_states)
    log_x_l, log_u_l = [x_curr], []

    for k in range(steps):
        try:
            u_opt, int_err = mpc_l.get_control(x_curr, int_err, X_ref[k:k+cfg.mpc_horizon+1], u_prev)
        except: 
            u_opt = u_prev
        x_curr = env.step(x_curr, u_opt, cfg.dt)
        log_x_l.append(x_curr); log_u_l.append(u_opt)
        u_prev = u_opt

    # ==========================================
    # 4. 全面绘图 (Enriched Visualization)
    # ==========================================
    print("[Phase 4] Generating Rich Plots...")
    min_len = min(len(log_x_k), len(log_x_l))
    t_plot = np.arange(0, cfg.sim_time + cfg.dt, cfg.dt)[:min_len]
    
    X_ref_cut = X_ref[:min_len]
    X_k_cut = np.array(log_x_k)[:min_len]
    X_l_cut = np.array(log_x_l)[:min_len]
    U_k_cut = np.array(log_u_k)[:min_len]
    U_l_cut = np.array(log_u_l)[:min_len]

    # [图1] 轨迹与控制 (时域)
    utils.plot_trajectory_and_control(
        t_plot, X_ref_cut, U_k_cut, X_k_cut, U_l_cut, X_l_cut, 
        cfg.results_dir
    )
    
    # [图2] 误差与性能指标 (RMSE & Energy) - 新增
    utils.plot_error_and_metrics(
        t_plot, X_ref_cut, U_k_cut, X_k_cut, U_l_cut, X_l_cut, 
        cfg.results_dir
    )
    
    # [图3] 相平面轨迹 (Phase Plane) - 新增
    # 展示 theta vs theta_dot 的闭环轨迹
    utils.plot_phase_plane_trajectory(
        X_ref_cut, X_k_cut, X_l_cut, 
        cfg.results_dir
    )
    
    # [图4] (倒立摆特有) 总能量变化曲线 - 物理意义分析
    # E = K + V = 1/6 * m * l^2 * theta_dot^2 + 0.5 * m * g * l * cos(theta)
    # (基于 dynamics.py 中 3g/2l 的系数推断是均匀杆模型)
    print("Generating Pendulum Energy Plot...")
    m, l, g = env.m, env.l, env.g
    
    def get_energy(x_traj):
        # x_traj shape: (N, 2) -> [theta, theta_dot]
        th = x_traj[:, 0]
        thdot = x_traj[:, 1]
        # 均匀杆转动惯量 I = m*l^2/3
        # 动能 K = 0.5 * I * w^2
        K = 0.5 * (m * l**2 / 3.0) * thdot**2
        # 势能 V (取转轴处为0势能面? 或者以最低点为0?)
        # dynamics方程暗示重力项是 1.5*g/l*sin(th)。
        # 这里为了可视化，我们画出 "相对势能" V = m * g * (l/2) * (1 - cos(theta)) (最低点为0)
        # 或者简单的 V = m * g * (l/2) * cos(theta) 看变化
        V = m * g * (l / 2.0) * (1 - np.cos(th))
        return K + V

    E_ref = get_energy(X_ref_cut)
    E_k = get_energy(X_k_cut)
    E_l = get_energy(X_l_cut)
    
    plt.figure(figsize=(10, 6))
    plt.plot(t_plot, E_ref, 'k--', label='Reference Energy', lw=2)
    plt.plot(t_plot, E_l, color='#117733', linestyle='-.', label='Linear MPC Energy', alpha=0.8)
    plt.plot(t_plot, E_k, color='#004488', linestyle='-', label='Koopman MPC Energy', lw=2)
    
    plt.xlabel('Time (s)', fontweight='bold')
    plt.ylabel('Total Energy (J)', fontweight='bold')
    plt.title('Pendulum Energy Evolution (Kinetic + Potential)', fontweight='bold', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    energy_plot_path = os.path.join(cfg.results_dir, "pendulum_energy_analysis.png")
    plt.savefig(energy_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Pendulum Simulation Done. All results saved to {cfg.results_dir}")
