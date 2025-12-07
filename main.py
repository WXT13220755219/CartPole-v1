import sys
# 确保能找到 common 和 systems 模块
sys.path.append(".") 

from systems.original import runner as original_runner
from systems.pendulum import runner as pendulum_runner
from systems.franka import runner as franka_runner

def main():
    print("Select System to Run:")
    print("1. Original (Benchmark System with Stabilization)")
    print("2. Pendulum (Inverted Pendulum with Sine Tracking)")
    print("3. Franka (7-DOF Robot Arm with PyBullet)")
    
    # 简单的交互选择，或者你可以直接注释掉不用的
    choice = input("Enter 1, 2 or 3: ")
    
    if choice == '1':
        original_runner.run()
    elif choice == '2':
        pendulum_runner.run()
    elif choice == '3':
        franka_runner.run()
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()