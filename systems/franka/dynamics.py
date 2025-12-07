import pybullet as p
import pybullet_data
import numpy as np
import time

class FrankaSystem:
    def __init__(self, gui=False):
        # 初始化 PyBullet
        self.client_id = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # 加载地面和机器人
        p.loadURDF("plane.urdf")
        # 使用 PyBullet 自带的 panda 模型
        # useFixedBase=True 保证基座固定
        self.robot_id = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
        
        self.num_joints = 7
        # PyBullet 中 Panda 的关节索引通常是 0-6 (对应 q1-q7)
        # 但需注意 panda.urdf 可能包含指关节，这里我们只控制前7个轴
        self.joint_indices = [0, 1, 2, 3, 4, 5, 6]
        
        # 启用力矩控制模式 (默认是位置控制，需要先关闭)
        for j in self.joint_indices:
            p.setJointMotorControl2(self.robot_id, j, controlMode=p.VELOCITY_CONTROL, force=0)

    def reset(self, state=None):
        """ 重置机器人状态 """
        if state is None:
            # 默认姿态 (Home)
            q = [0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.7]
            dq = [0.0] * 7
        else:
            q = state[:7]
            dq = state[7:]
            
        for i, j_idx in enumerate(self.joint_indices):
            p.resetJointState(self.robot_id, j_idx, targetValue=q[i], targetVelocity=dq[i])
        
        return self.get_state()

    def get_state(self):
        """ 获取当前 (q, dq) """
        joint_states = p.getJointStates(self.robot_id, self.joint_indices)
        # joint_states[i] = (pos, vel, reaction_forces, applied_torque)
        q = [state[0] for state in joint_states]
        dq = [state[1] for state in joint_states]
        return np.array(q + dq)

    def step(self, x_current, u_current, dt):
        """ 
        执行一步仿真 
        x_current: 当前状态 (仅用于校验，PyBullet内部维护状态)
        u_current: 7维力矩输入
        dt: 这里的 dt 主要用于记录，实际步长由 p.setTimeStep 决定
        """
        # 1. 设置仿真步长 (保持与 config 一致)
        p.setTimeStep(dt)
        
        # 2. 应用力矩控制
        # 注意：PyBullet 的 setJointMotorControl2 在 TORQUE_CONTROL 模式下
        # force 参数即为力矩
        for i, j_idx in enumerate(self.joint_indices):
            torque = np.clip(u_current[i], -87, 87) # 简单的限幅保护
            p.setJointMotorControl2(self.robot_id, j_idx, 
                                    controlMode=p.TORQUE_CONTROL, 
                                    force=torque)
        
        # 3. 物理引擎步进
        p.stepSimulation()
        
        # 4. 返回新状态
        return self.get_state()

    def get_jacobian(self, x, u):
        # 占位符：SINDy-KMPC 不需要物理模型的 Jacobian
        # 如果需要 LinearMPC，这里需要用数值差分法求 A, B
        return None, None

    def close(self):
        p.disconnect(self.client_id)