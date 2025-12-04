import pybullet as pb
import pybullet_data
import numpy as np
import time
import os

class FrankaSystem:
    def __init__(self, render=False):
        # 初始化 PyBullet
        if render:
            self.client = pb.connect(pb.GUI)
        else:
            self.client = pb.connect(pb.DIRECT)
            
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        pb.loadURDF('plane.urdf')
        
        # 尝试加载 Franka Panda 模型
        # 优先查找本地路径，如果没有则使用 PyBullet 自带的
        local_urdf = "./franka/franka_description/robots/franka_panda.urdf"
        if os.path.exists(local_urdf):
            print(f"Loading local URDF: {local_urdf}")
            self.robot = pb.loadURDF(local_urdf, [0.,0.,0.], useFixedBase=1)
        else:
            print("Local URDF not found, using PyBullet internal Panda model.")
            self.robot = pb.loadURDF("franka_panda/panda.urdf", [0,0,0], useFixedBase=1)
            
        pb.setGravity(0, 0, -9.81)
        
        # 确定末端执行器 Link ID (通常是 11 或 7，取决于URDF)
        # 这里我们简单遍历一下找到叫 'panda_hand' 或类似的 link，或者直接用索引
        self.ee_id = 11 # PyBullet 自带 panda 的 EE 是 11
        for i in range(pb.getNumJoints(self.robot)):
            info = pb.getJointInfo(self.robot, i)
            link_name = info[12].decode("utf-8")
            if 'hand' in link_name or 'ee' in link_name:
                self.ee_id = i
                
        self.u_max = 50.0

    def reset_to_state(self, x):
        """
        强制将 PyBullet 物理引擎重置为向量 x 指定的状态
        x: [EE_pos(3), Joint_pos(7), Joint_vel(7)]
        注意：EE_pos 是由 Joint_pos 决定的，这里我们只重置关节
        """
        joint_pos = x[3:10]
        joint_vel = x[10:17]
        
        index = 0
        for i in range(pb.getNumJoints(self.robot)):
            # 只有非固定关节才需要重置 (Franka有7个可控关节)
            info = pb.getJointInfo(self.robot, i)
            q_index = info[3]
            if q_index > -1: # 是可动关节
                if index < 7:
                    pb.resetJointState(self.robot, i, joint_pos[index], joint_vel[index])
                    index += 1

    def get_obs(self):
        """获取当前观测向量 (17维)"""
        # 1. 获取关节状态
        jnt_pos = []
        jnt_vel = []
        
        for i in range(pb.getNumJoints(self.robot)):
            info = pb.getJointInfo(self.robot, i)
            if info[3] > -1 and len(jnt_pos) < 7: # 收集前7个关节
                state = pb.getJointState(self.robot, i)
                jnt_pos.append(state[0])
                jnt_vel.append(state[1])
        
        # 2. 获取末端执行器状态 (EE Position)
        ee_state = pb.getLinkState(self.robot, self.ee_id)
        ee_pos = ee_state[0] # (x, y, z)
        
        # 拼接: EE(3) + Pos(7) + Vel(7) = 17 dims
        return np.concatenate([ee_pos, jnt_pos, jnt_vel])

    def step(self, x_current, u_current, dt):
        """
        执行一步仿真：
        SINDy-MPC 需要形如 x_next = f(x, u) 的函数。
        """
        # 1. 强制设置当前状态 (确保无状态函数的假设成立)
        self.reset_to_state(x_current)
        
        # 2. 应用控制 (力矩控制)
        u = np.clip(u_current, -self.u_max, self.u_max)
        
        # 找到那7个可控关节的索引
        control_indices = []
        for i in range(pb.getNumJoints(self.robot)):
            info = pb.getJointInfo(self.robot, i)
            if info[3] > -1 and len(control_indices) < 7:
                control_indices.append(i)

        pb.setJointMotorControlArray(
            self.robot, control_indices,
            pb.TORQUE_CONTROL, forces=u
        )
        
        # 3. 步进仿真
        # PyBullet 默认 timestep 通常是 1/240
        sim_step = 1./240.
        num_steps = int(dt / sim_step)
        if num_steps < 1: num_steps = 1
        
        for _ in range(num_steps):
            pb.stepSimulation()
            
        # 4. 返回新状态
        return self.get_obs()
    
    def close(self):
        pb.disconnect()

    def get_jacobian(self, x, u):
        # 机械臂的解析线性化极其复杂，这里不提供 LinearMPC 支持
        # SINDy-KMPC 不需要这个函数
        return None, None