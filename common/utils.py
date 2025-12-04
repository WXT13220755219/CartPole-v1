import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.patches as mpatches

# =========================================================
# 0. SCI 全局绘图风格配置
# =========================================================
def set_sci_style():
    plt.style.use('default')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 15
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['grid.alpha'] = 0.4
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.pad_inches'] = 0.1

set_sci_style()

def generate_figure_8_traj(t_seq, scale=1.5, omega=0.5):
    x_ref = scale * np.sin(omega * t_seq)
    y_ref = scale * np.sin(omega * t_seq) * np.cos(omega * t_seq)
    return np.vstack([x_ref, y_ref]).T

# =========================================================
# 1. 热力图绘制
# =========================================================

def plot_separated_AB_heatmaps(A, B, save_dir, feature_names, input_names):
    n_features = len(feature_names) 
    n_inputs = len(input_names)     
    
    width_ratios = [n_features, n_inputs + 1.2] 
    fig_height = max(8, n_features * 0.6)
    total_cols = n_features + n_inputs
    fig_width = fig_height * (total_cols / n_features) * 1.4 
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, fig_height), 
                                   gridspec_kw={'width_ratios': width_ratios},
                                   constrained_layout=True)
    
    threshold = 1e-3
    max_val = max(np.max(np.abs(A)), np.max(np.abs(B)))
    
    mask_A = np.abs(A) < threshold
    sns.heatmap(np.abs(A), annot=A, cmap="Reds", mask=mask_A, 
                vmin=0, vmax=max_val,
                center=None, robust=True, fmt=".2f",
                linewidths=0.5, linecolor='white',
                cbar=False, 
                xticklabels=feature_names, yticklabels=feature_names,
                annot_kws={"size": 10}, square=True, ax=ax1)
    ax1.set_title("Matrix A: State Transition", fontweight='bold', pad=15)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    
    mask_B = np.abs(B) < threshold
    sns.heatmap(np.abs(B), annot=B, cmap="Reds", mask=mask_B,
                vmin=0, vmax=max_val,
                center=None, robust=True, fmt=".2f",
                linewidths=0.5, linecolor='white',
                cbar=True, 
                cbar_kws={'label': '|Value|', 'fraction': 0.15, 'pad': 0.05},
                xticklabels=input_names, yticklabels=feature_names,
                annot_kws={"size": 10}, square=True, ax=ax2)
    ax2.set_title("Matrix B: Control Input", fontweight='bold', pad=15)
    ax2.set_yticks([]) 
    
    save_path = os.path.join(save_dir, "heatmap_1_separated_AB.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_combined_structure_heatmap(A, B, save_dir, feature_names, input_names):
    K_full = np.hstack([A, B])
    all_source_names = list(feature_names) + list(input_names)
    
    n_rows, n_cols = K_full.shape
    fig_width = max(16, n_cols * 0.8)
    fig_height = max(10, n_rows * 0.6)
    
    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
    
    threshold = 1e-3
    mask = np.abs(K_full) < threshold
    
    ax = sns.heatmap(np.abs(K_full), annot=K_full, cmap="Reds", mask=mask,
                     vmin=0, vmax=1.2, 
                     center=None, robust=True, fmt=".3f",
                     linewidths=1.0, linecolor='white',
                     annot_kws={"size": 11},
                     xticklabels=all_source_names, yticklabels=feature_names,
                     cbar_kws={'label': '|Value|', 'shrink': 0.8},
                     square=True)
    
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    plt.xticks(rotation=45, ha='left', fontsize=12, fontweight='bold')
    plt.title("SINDy Identified Sparse Structure", fontsize=18, fontweight='bold', pad=30, y=1.05)
    plt.axvline(x=A.shape[1], color='#333333', linestyle='--', linewidth=2.5, alpha=0.8)
    
    save_path = os.path.join(save_dir, "heatmap_2_combined_structure.png")
    plt.savefig(save_path, dpi=300)
    plt.close()


# =========================================================
# 2. 对比绘图 (已修复 IndexError)
# =========================================================

def plot_trajectory_and_control(t, x_ref, u_log_k, x_log_k, u_log_l, x_log_l, save_dir):
    """ [图3] 轨迹与控制 (自适应输入维度) """
    c_ref = 'black'        
    c_koop = '#004488'     
    c_lin = '#117733'      
    
    # === 关键修改：获取实际输入维度 ===
    # 假设 u_log 的 shape 是 (N, n_inputs)
    inputs_dim = u_log_k.shape[1]
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    
    def plot_track(ax, time, ref, dat_k, dat_l, y_lab, title):
        ax.plot(time, ref, color=c_ref, ls='--', lw=2.5, alpha=0.7, label='Reference')
        ax.plot(time, dat_l, color=c_lin, ls='-.', lw=2.0, alpha=0.8, label='Linear MPC')
        ax.plot(time, dat_k, color=c_koop, ls='-', lw=2.2, alpha=0.9, label='Koopman MPC')
        ax.set_ylabel(y_lab, fontweight='bold')
        ax.set_title(title, fontweight='bold', fontsize=14)
        ax.grid(True, which='both', ls=':', alpha=0.5)
        ax.legend(loc='lower right', framealpha=0.9, edgecolor='gray')

    def plot_ctrl(ax, time, dat_k, dat_l, y_lab, title):
        ax.step(time, dat_l, where='post', color=c_lin, ls='-.', lw=2.0, alpha=0.8, label='Linear MPC')
        ax.step(time, dat_k, where='post', color=c_koop, ls='-', lw=2.2, alpha=0.9, label='Koopman MPC')
        ax.set_ylabel(y_lab, fontweight='bold')
        ax.set_title(title, fontweight='bold', fontsize=14)
        ax.grid(True, which='both', ls=':', alpha=0.5)
        ax.legend(loc='upper right', framealpha=0.9, edgecolor='gray')

    # 1. 绘制状态 (x1, x2)
    plot_track(axs[0, 0], t, x_ref[:, 0], x_log_k[:, 0], x_log_l[:, 0], 
               "Position $x_1$ (m)", "State $x_1$ Tracking")
    plot_track(axs[0, 1], t, x_ref[:, 1], x_log_k[:, 1], x_log_l[:, 1], 
               "Position $x_2$ (m)", "State $x_2$ Tracking")
    
    # 2. 绘制控制 u1 (始终存在)
    plot_ctrl(axs[1, 0], t[:-1], u_log_k[:, 0], u_log_l[:, 0], 
              "Control $u_1$ (N)", "Control Input $u_1$")
    axs[1, 0].set_xlabel("Time (s)", fontweight='bold')

    # 3. 绘制控制 u2 (仅当输入 > 1 时)
    if inputs_dim > 1:
        plot_ctrl(axs[1, 1], t[:-1], u_log_k[:, 1], u_log_l[:, 1], 
                  "Control $u_2$ (N)", "Control Input $u_2$")
        axs[1, 1].set_xlabel("Time (s)", fontweight='bold')
    else:
        # 单输入系统，隐藏第四张图
        axs[1, 1].axis('off')
        axs[1, 1].text(0.5, 0.5, "Single Input System\n(No $u_2$)", 
                       ha='center', va='center', fontsize=16, color='gray', alpha=0.5)

    save_path = os.path.join(save_dir, "comparison_1_traj_control.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_error_and_metrics(t, x_ref, u_log_k, x_log_k, u_log_l, x_log_l, save_dir):
    """ [图4] 误差与指标 """
    c_koop = '#004488' 
    c_lin = '#117733'  
    
    err_k = np.linalg.norm(x_log_k - x_ref, axis=1)
    err_l = np.linalg.norm(x_log_l - x_ref, axis=1)
    mean_err_k, mean_err_l = np.mean(err_k), np.mean(err_l)
    eng_k, eng_l = np.sum(u_log_k**2), np.sum(u_log_l**2)
    
    fig = plt.figure(figsize=(12, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 0.7])
    
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(t, err_l, color=c_lin, ls='-.', lw=2.0, alpha=0.8, label='Linear MPC Error')
    ax1.plot(t, err_k, color=c_koop, ls='-', lw=2.2, label='Koopman MPC Error')
    ax1.fill_between(t, 0, err_k, color=c_koop, alpha=0.1) 
    ax1.fill_between(t, 0, err_l, color=c_lin, alpha=0.05)
    
    ax1.set_ylabel("Error Norm $||x - x_{ref}||_2$", fontweight='bold')
    ax1.set_xlabel("Time (s)", fontweight='bold')
    ax1.set_title("Tracking Error Evolution", fontweight='bold', fontsize=15)
    ax1.grid(True, ls=':', alpha=0.5)
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.set_xlim([0, t[-1]])
    ax1.set_ylim(bottom=0)
    
    ax2 = fig.add_subplot(gs[1])
    categories = ['RMSE (Avg Error)', r'Control Energy ($\Sigma u^2$)']
    y_pos = np.arange(len(categories))
    height = 0.35
    
    val_k = [mean_err_k, eng_k]
    val_l = [mean_err_l, eng_l]
    norm_k = [k/l for k, l in zip(val_k, val_l)]
    norm_l = [1.0, 1.0]
    
    ax2.barh(y_pos - height/2 - 0.02, norm_l, height, color=c_lin, alpha=0.6, label='Linear MPC (Baseline)')
    ax2.barh(y_pos + height/2 + 0.02, norm_k, height, color=c_koop, alpha=0.8, label='Koopman MPC (Proposed)')
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(categories, fontsize=12, fontweight='bold')
    ax2.set_xlabel("Normalized Metric (Lower is Better)", fontweight='bold')
    ax2.set_title("Performance Comparison (Normalized)", fontweight='bold', fontsize=15)
    ax2.legend(loc='lower right')
    ax2.set_xlim(0, 1.35)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.grid(axis='x', ls='--', alpha=0.3)
    
    def add_labels(vals, is_k=False, y_offset=0):
        for i, val in enumerate(vals):
            if val > 100: txt = f"{val:.1f}"
            else: txt = f"{val:.4f}"
            display_val = norm_k[i] if is_k else norm_l[i]
            x_pos = display_val + 0.02
            ax2.text(x_pos, y_pos[i] + y_offset, txt, va='center', fontsize=11)
            if is_k:
                imp = (1 - display_val) * 100
                color = '#117733' if imp > 0 else '#d62728' 
                sign = "-" if imp > 0 else "+"
                ax2.text(x_pos + 0.12, y_pos[i] + y_offset, f"({sign}{abs(imp):.1f}%)", 
                         va='center', fontsize=11, fontweight='bold', color=color)

    add_labels(val_l, is_k=False, y_offset = -height/2 - 0.02)
    add_labels(val_k, is_k=True, y_offset = height/2 + 0.02)
    ax2.invert_yaxis()
    
    save_path = os.path.join(save_dir, "comparison_2_metrics.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_phase_plane_trajectory(x_ref, x_log_k, x_log_l, save_dir):
    """ [图5] 相平面轨迹对比 """
    c_ref = 'black'        
    c_koop = '#004488'     
    c_lin = '#117733'
    
    plt.figure(figsize=(10, 8), constrained_layout=True)
    plt.plot(x_ref[:, 0], x_ref[:, 1], color=c_ref, ls='--', lw=3.0, alpha=0.6, label='Reference')
    plt.plot(x_log_l[:, 0], x_log_l[:, 1], color=c_lin, ls='-.', lw=2.0, alpha=0.8, label='Linear MPC')
    plt.plot(x_log_k[:, 0], x_log_k[:, 1], color=c_koop, ls='-', lw=2.5, alpha=0.9, label='Koopman MPC')
    
    plt.scatter(x_ref[0, 0], x_ref[0, 1], color='green', s=150, zorder=10, edgecolors='black', label='Start')
    plt.scatter(x_log_k[-1, 0], x_log_k[-1, 1], color=c_koop, marker='X', s=150, zorder=10, label='End (Koopman)')
    
    plt.xlabel("State $x_1$", fontweight='bold')
    plt.ylabel("State $x_2$", fontweight='bold')
    plt.title("Phase Plane Trajectory ($x_1$ vs $x_2$)", fontweight='bold', fontsize=16)
    plt.grid(True, ls=':', alpha=0.5)
    plt.axis('equal') 
    plt.legend(loc='best', framealpha=0.9, edgecolor='gray')
    
    save_path = os.path.join(save_dir, "comparison_3_phase_plane.png")
    plt.savefig(save_path, dpi=300)
    print(f"[Result] Plots saved to {save_dir}")
    plt.close()