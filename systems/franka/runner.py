import numpy as np
import matplotlib.pyplot as plt
import os

from common.koopman import SindyKoopman
from common.mpc import KoopmanMPC
import common.utils as utils
from .config import FrankaConfig
from .dynamics import FrankaSystem

def run():
    print("=== Running Franka Panda System (SINDy-KMPC) ===")
    print("提示: 17维状态空间的 SINDy 训练和 MPC 求解可能较慢，请耐心等待。")
    
    # 1. 初始化
    cfg = FrankaConfig(exp_note="pose_hold")
    # render=True 可以看到 PyBullet 窗口，想加速数据采集可以设为 False
    env = FrankaSystem(render=True) 
    kp_model = SindyKoopman(cfg)

    # ==========================================
    # 2. 数据采集 (Data Collection)
    # ==========================================
    print(f"[Phase 1] Collecting {cfg.data_samples} samples...")
    X, U, Xn = [], [], []
    
    # 定义一个"家"的位置作为重置中心
    home_joints = np.array([0., -0.78, 0., -2.35, 0., 1.57, 0.78]) # Panda 经典初始位
    
    # 先 reset 一次获取合法的 x 向量结构
    env.reset_to_state(np.concatenate([np.zeros(3), home_joints, np.zeros(7)]))
    x = env.get_obs()
    
    for i in range(cfg.data_samples):
        # 策略：以一定概率随机重置，防止机械臂跑飞或卡死
        if i % 50 == 0:
            noise = np.random.uniform(-0.5, 0.5, size=7)
            q_new = home_joints + noise
            # 简单的限位保护
            q_new = np.clip(q_new, cfg.joint_low, cfg.joint_high)
            env.reset_to_state(np.concatenate([np.zeros(3), q_new, np.zeros(7)]))
            x = env.get_obs()

        # 随机控制输入 (Excitation)
        u = np.random.uniform(-20, 20, size=(cfg.n_inputs,))
        
        xn = env.step(x, u, cfg.dt)
        
        X.append(x); U.append(u); Xn.append(xn)
        x = xn
        
        if (i+1) % 1000 == 0:
            print(f"  Sample {i+1}/{cfg.data_samples} collected.")
    
    # ==========================================
    # 3. 训练 Koopman 模型
    # ==========================================
    print("[Phase 2] Training SINDy Model...")
    kp_model.fit(np.array(X), np.array(U), np.array(Xn))
    
    feat_names = kp_model.get_feature_names()
    print(f"  Learned {len(feat_names)} features (Lifted state dimension).")

    # ==========================================
    # 4. 闭环控制：姿态保持 (Pose Holding)
    # ==========================================
    print("[Phase 3] KMPC Control Evaluation...")
    
    # 设定目标：回到 Home Pose 并且保持静止
    env.reset_to_state(np.concatenate([np.zeros(3), home_joints, np.zeros(7)]))
    target_x = env.get_obs() # 获取真实的 Home State
    
    # 构造参考轨迹 (全是目标状态)
    steps = int(cfg.sim_time / cfg.dt)
    X_ref = np.tile(target_x, (steps + cfg.mpc_horizon + 1, 1))
    
    # 提升参考轨迹
    # 注意：Z_ref 可能很大，如果内存不够可以只在循环里实时 lift
    Z_ref = kp_model.lift(X_ref) 
    
    # 初始化 MPC
    mpc_k = KoopmanMPC(cfg, kp_model)
    
    # 将机械臂设置到一个偏离的位置 (Initial Disturbance)
    disturbed_joints = home_joints + np.array([0.2, -0.2, 0.1, 0.1, 0, 0, 0])
    env.reset_to_state(np.concatenate([np.zeros(3), disturbed_joints, np.zeros(7)]))
    x_curr = env.get_obs()
    
    u_prev = np.zeros(cfg.n_inputs)
    int_err = np.zeros(cfg.n_states) # 实际上 config里积分增益是0，这项没用
    
    log_x = [x_curr]
    log_u = []
    
    print("  Running MPC loop...")
    for k in range(steps):
        # 1. 当前状态提升
        z_curr = kp_model.lift(x_curr).flatten()
        
        # 2. 求解 MPC
        try:
            # 这里的 MPC 求解器可能会报 "Solved/Inaccurate" 甚至失败
            # 因为 17维 x 200维特征 的 QP 问题规模较大
            u_opt, int_err = mpc_k.get_control(
                z_curr, int_err, 
                Z_ref[k:k+cfg.mpc_horizon+1], 
                X_ref[k:k+cfg.mpc_horizon+1], 
                u_prev
            )
        except Exception as e:
            print(f"  [MPC Error at step {k}]: {e}")
            u_opt = np.zeros(cfg.n_inputs) # 故障回退：零力矩
            
        # 3. 施加控制
        x_curr = env.step(x_curr, u_opt, cfg.dt)
        
        log_x.append(x_curr)
        log_u.append(u_opt)
        u_prev = u_opt
        
        # 打印 EE 误差距离
        ee_dist = np.linalg.norm(x_curr[:3] - target_x[:3])
        if k % 10 == 0:
            print(f"  Step {k}/{steps} | EE Error: {ee_dist:.4f} m")
            
    env.close()

    # ==========================================
    # 5. 绘图结果
    # ==========================================
    print("Plotting results...")
    log_x = np.array(log_x)
    log_u = np.array(log_u)
    
    # 绘制末端执行器位置 (XYZ)
    plt.figure(figsize=(10, 6))
    dims = ['x', 'y', 'z']
    for i in range(3):
        plt.subplot(3, 1, i+1)
        plt.plot(log_x[:, i], label=f'Actual {dims[i]}')
        plt.axhline(target_x[i], color='r', linestyle='--', label='Target')
        plt.ylabel(f'{dims[i]} (m)')
        if i == 0: plt.legend()
    plt.xlabel('Steps')
    plt.suptitle('Franka End-Effector Position Control (SINDy-KMPC)')
    plt.savefig(os.path.join(cfg.results_dir, "franka_ee_traj.png"))
    
    # 绘制关节角度 (前3个主要关节)
    plt.figure(figsize=(10, 6))
    for i in range(3):
        plt.subplot(3, 1, i+1)
        # 状态索引: 0-2是EE, 3-9是JointPos
        idx = 3 + i
        plt.plot(log_x[:, idx], label=f'Joint {i+1}')
        plt.axhline(target_x[idx], color='r', linestyle='--', label='Target')
        plt.ylabel('Rad')
    plt.xlabel('Steps')
    plt.suptitle('Joint Angles (First 3 Joints)')
    plt.savefig(os.path.join(cfg.results_dir, "franka_joint_traj.png"))
    
    print(f"Done! Results saved in {cfg.results_dir}")

if __name__ == "__main__":
    run()