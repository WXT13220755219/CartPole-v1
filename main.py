import sys
# 确保能找到 common 和 systems 模块
sys.path.append(".") 

from systems.pendulum import runner as pendulum_runner
from systems.original import runner as original_runner

def main():
    print("Select System to Run:")
    print("1. Pendulum (Inverted Pendulum with Sine Tracking)")
    print("2. Original (Benchmark System with Stabilization)")
    
    # 简单的交互选择，或者你可以直接注释掉不用的
    choice = input("Enter 1 or 2: ")
    
    if choice == '1':
        pendulum_runner.run()
    elif choice == '2':
        original_runner.run()
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()