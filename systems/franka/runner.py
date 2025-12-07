import numpy as np
import time
from common.koopman import SindyKoopman
from common.mpc import KoopmanMPC
import common.utils as utils
from .config import FrankaConfig
from .dynamics import FrankaSystem

def run():
    print("=== Running Franka Emika Panda System (SINDy-KMPC) ===")
    
    # 1. 初始化配置与环境
    cfg = FrankaConfig(exp_note="joint_holding")
    # 训练阶段使用 DIRECT 模式加速，测试阶段可以用 GUI 模式 (修改 dynamics.py 接口可支持)
    env = FrankaSystem(gui=False) 
    kp_model = SindyKoopman(cfg)

    # ==========================================
    # Phase 1: 数据采集 (Data Collection)
    # ==========================================
    print(f"[Phase 1] Collecting Data ({cfg.data_samples} samples)...")
    X, U, Xn = [], [], []
    
    # 初始复位
    x = env.reset()
    
    # 进度显示辅助
    print_interval = cfg.data_samples // 10
    
    for i in range(cfg.data_samples):
        # 激励策略：叠加不同频率的正弦波 + 噪声，以充分激发系统动力学特性
        u = np.zeros(cfg.n_inputs)
        for j in range(cfg.n_inputs):
            # 每个关节给不同的频率
            freq = 0.5 + 0.2 * j
            u[j] = 40.0 * np.sin(freq * i * cfg.dt) + np.random.normal(0, 5.0)
        
        # 执行一步
        u = np.clip(u, -80, 80)
        
        xn = env.step(x, u, cfg.dt)
        
        X.append(x)
        U.append(u)
        Xn.append(xn)
        
        # 定期重置，防止机器人跑飞或陷入死锁
        # 判据：每 200 步 或 速度过大
        if i % 200 == 0 or np.max(np.abs(xn[7:])) > 5.0:
            # 随机复位到一个安全范围内
            rand_q = np.random.uniform(-0.5, 0.5, size=7) + np.array([0, -0.5, 0, -2.0, 0, 1.5, 0])
            rand_dq = np.zeros(7)
            x = env.reset(np.concatenate([rand_q, rand_dq]))
        else:
            x = xn
            
        if (i+1) % print_interval == 0:
            print(f"  -> Collected {i+1}/{cfg.data_samples} samples")

    # ==========================================
    # Phase 2: 模型训练 (Model Training)
    # ==========================================
    print("[Phase 2] Training SINDy Model...")
    # 转换为数组
    X_arr, U_arr, Xn_arr = np.array(X), np.array(U), np.array(Xn)
    
    # 训练
    kp_model.fit(X_arr, U_arr, Xn_arr)
    
    # 可视化结构 (仅当特征数不过大时)
    # 14状态 + 7输入，如果 poly_order=1，热力图还比较清晰
    try:
        print("  -> Generating structure heatmap...")
        feat_names = kp_model.get_feature_names()
        inp_names = [f"tau_{i+1}" for i in range(cfg.n_inputs)]
        utils.plot_combined_structure_heatmap(
            kp_model.A, kp_model.B, 
            cfg.results_dir, 
            feat_names, 
            inp_names
        )
    except Exception as e:
        print(f"  [Warn] Heatmap generation skipped: {e}")

    # ==========================================
    # Phase 3: 闭环控制 (Control Evaluation)
    # ==========================================
    print("[Phase 3] MPC Control Evaluation...")
    
    # 为了演示，重新开启一个带 GUI 的环境 (可选，如果想看动画)
    env.close()
    env = FrankaSystem(gui=False) # 如果在服务器上跑，保持 False
    
    steps = int(cfg.sim_time / cfg.dt)
    
    # --- 任务：状态保持 (Regulation) ---
    # 目标：保持在初始位置 [0, -0.5, 0, -2.0, 0, 1.5, 0]
    # 这是一个非零力矩平衡点，需要 MPC 自动计算重力补偿
    
    # 定义参考轨迹 (定点)
    target_q = np.array([0.5, -0.5, 0.0, -1.5, 0.0, 1.5, 0.0]) # 稍微移动一点位置作为目标
    target_dq = np.zeros(7)
    x_ref_single = np.concatenate([target_q, target_dq])
    
    # 构造整个时间视窗的参考轨迹
    X_ref = np.tile(x_ref_single, (steps + cfg.mpc_horizon + 1, 1))
    Z_ref = kp_model.lift(X_ref)
    
    # MPC 初始化
    mpc_k = KoopmanMPC(cfg, kp_model)
    
    # 初始状态设置 (从零位开始，去追踪 target_q)
    x_curr = env.reset() # reset 到默认 Home
    u_prev = np.zeros(cfg.n_inputs)
    int_err = np.zeros(cfg.n_states)
    
    log_x = [x_curr]
    log_u = []
    
    print(f"  -> Target Joint 1: {target_q[0]:.2f}, Start Joint 1: {x_curr[0]:.2f}")

    start_time = time.time()
    for k in range(steps):
        # 1. 提升当前状态
        z_curr = kp_model.lift(x_curr).flatten()
        
        # 2. MPC 求解
        try:
            # 提取未来 horizon 的参考
            z_ref_h = Z_ref[k : k + cfg.mpc_horizon + 1]
            x_ref_h = X_ref[k : k + cfg.mpc_horizon + 1]
            
            u_opt, int_err = mpc_k.get_control(z_curr, int_err, z_ref_h, x_ref_h, u_prev)
        except Exception as e:
            print(f"  [Error] MPC failed at step {k}: {e}")
            break
            
        # 3. 执行控制
        x_curr = env.step(x_curr, u_opt, cfg.dt)
        
        # 4. 记录
        log_x.append(x_curr)
        log_u.append(u_opt)
        u_prev = u_opt
        
        if k % 50 == 0:
            # 简单的进度条
            print(f"    Step {k}/{steps}, Err(q1)={x_curr[0]-target_q[0]:.4f}")

    print(f"  -> Simulation finished in {time.time()-start_time:.2f}s")

    # ==========================================
    # Phase 4: 结果绘图
    # ==========================================
    # 由于维度太高 (14维)，我们借用 utils 只绘制前 2 个关节 (q1, q2) 和前 2 个力矩
    # 构造伪造的 LinearMPC 数据 (全0) 用于占位，以复用 plot_trajectory_and_control 函数
    
    log_x = np.array(log_x)
    log_u = np.array(log_u)
    
    # 截取前两维
    x_dim_plot = 2 
    u_dim_plot = 2
    
    # 构造绘图所需的时间轴
    t_plot = np.arange(len(log_x)) * cfg.dt
    
    # 提取用于绘图的数据切片
    # 这里的 hack 是：把 14 维数据切成 2 维传给绘图函数
    x_ref_plot = X_ref[:len(log_x), :x_dim_plot]
    x_k_plot   = log_x[:, :x_dim_plot]
    u_k_plot   = log_u[:, :u_dim_plot]
    
    # 伪造对比数据 (全零)，因为没有 Linear MPC
    x_l_dummy = np.zeros_like(x_k_plot)
    u_l_dummy = np.zeros_like(u_k_plot)
    
    print("  -> Saving plots...")
    utils.plot_trajectory_and_control(
        t_plot, 
        x_ref_plot, 
        u_k_plot, x_k_plot,   # Koopman 数据
        u_l_dummy, x_l_dummy, # Linear 数据 (空)
        cfg.results_dir
    )
    
    print(f"Franka Experiment Done. Results saved to {cfg.results_dir}")
    env.close()