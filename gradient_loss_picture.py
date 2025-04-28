import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import torch
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']


def plot_all_metrics(result_dir):
    # 创建专用子文件夹
    plot_dir = os.path.join(result_dir, 'training_metrics')
    os.makedirs(plot_dir, exist_ok=True)

    with open(os.path.join(result_dir, 'cut_loss.pkl'), 'rb') as f:
        data = pickle.load(f)

    # ==================== 1. Actor梯度范数 ====================
    plt.figure(figsize=(10, 6))
    for agent_id, gradients in data['actor_gradient'].items():
        if len(gradients) > 0:  # 过滤空数据
            gradients = [g.item() if torch.is_tensor(g) else g for g in gradients]
            plt.plot(gradients, label=agent_id, alpha=0.7)
    plt.title('Actor Gradient Norms')
    plt.xlabel('Training Steps')
    plt.ylabel('Gradient Norm')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(plot_dir, '1_actor_gradients.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ==================== 2. Critic梯度范数 ====================
    plt.figure(figsize=(10, 6))
    for agent_id, gradients in data['critic_gradient'].items():
        if len(gradients) > 0:
            gradients = [g.item() if torch.is_tensor(g) else g for g in gradients]
            plt.plot(gradients, label=agent_id, alpha=0.7)
    plt.title('Critic Gradient Norms')
    plt.xlabel('Training Steps')
    plt.ylabel('Gradient Norm')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(plot_dir, '2_critic_gradients.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ==================== 3. Mixing网络总梯度 ====================
    if len(data['mix_gradient']) > 0:
        plt.figure(figsize=(10, 6))
        mix_grads = [entry[0].item() if torch.is_tensor(entry[0]) else entry[0] for entry in data['mix_gradient']]
        plt.plot(mix_grads, color='purple', alpha=0.7)
        plt.title('QMIX Network Gradient Norm')
        plt.xlabel('Training Steps')
        plt.ylabel('Gradient Norm')
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(plot_dir, '3_qmix_gradients.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # ==================== 4. 追逐者局部Q梯度 ====================
    if len(data['mix_gradient']) > 0 and isinstance(data['mix_gradient'][0][1], dict):
        plt.figure(figsize=(10, 6))
        all_adv_ids = list(data['mix_gradient'][0][1].keys())
        color_map = plt.cm.get_cmap('tab10', len(all_adv_ids))

        # 为每个追逐者创建单独曲线
        for idx, adv_id in enumerate(all_adv_ids):
            gradients = []
            for entry in data['mix_gradient']:
                grad = entry[1].get(adv_id, np.nan)
                gradients.append(grad.item() if torch.is_tensor(grad) else grad)
            plt.plot(gradients, color=color_map(idx), label=adv_id, alpha=0.7)

        plt.title('Adversary Local Q Gradients')
        plt.xlabel('Training Steps')
        plt.ylabel('Gradient Norm')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(plot_dir, '4_adv_local_gradients.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # ==================== 5. Critic损失 ====================
    plt.figure(figsize=(10, 6))
    for agent_id, losses in data['critic_loss'].items():
        if len(losses) > 0:
            plt.plot(losses, label=agent_id, alpha=0.7)
    plt.title('Global Critic Loss')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(plot_dir, '5_critic_loss.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ==================== 6. Actor损失 ====================
    plt.figure(figsize=(10, 6))
    for agent_id, losses in data['actor_loss'].items():
        if len(losses) > 0:
            plt.plot(losses, label=agent_id, alpha=0.7)
    plt.title('Actor Loss')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(plot_dir, '6_actor_loss.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ==================== 7. QMIX损失 ====================
    if len(data['mix_loss']) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(data['mix_loss'], color='darkorange', alpha=0.7)
        plt.title('QMIX Network Loss')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(plot_dir, '7_qmix_loss.png'), dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    result_dir = './results/simple_tag_v2/5'  # 修改为你的实际路径
    plot_all_metrics(result_dir)