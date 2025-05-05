import os
import glob
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from main import get_env
from MADDPG import MADDPG

# 配置参数
BASE_DIR = './results/compare_suanfa'
EPISODE_NUM = 1000
EPISODE_LENGTH = 25


def evaluate_model(model_path, env_name):
    """评估单个模型并返回平均合围次数"""
    env, dim_info, num_good, num_adversaries = get_env(env_name, EPISODE_LENGTH)
    maddpg = MADDPG.load(dim_info, model_path, num_good, num_adversaries)

    total_cooperation = 0
    sum_total_reward = 0
    for _ in tqdm(range(EPISODE_NUM), desc="Evaluating"):
        states = env.reset()

        for _ in range(EPISODE_LENGTH):
            actions = maddpg.select_action(states)
            next_states, _, total_reward, _, _, cooperation_times = env.step(actions)  # 接收新返回值

            total_cooperation += cooperation_times  # 直接累加环境返回的合围次数
            sum_total_reward += total_reward
            states = next_states


    env.close()
    return total_cooperation / (EPISODE_NUM * EPISODE_LENGTH), sum_total_reward / (EPISODE_NUM * EPISODE_LENGTH) # 平均每step合围概率


def main():
    results = {}

    for algorithm in os.listdir(BASE_DIR):
        algorithm_path = os.path.join(BASE_DIR, algorithm)
        if not os.path.isdir(algorithm_path):
            continue


        cooperation_rates = []
        cooperation_total_rewards = []
        for exp_dir in glob.glob(os.path.join(algorithm_path, '*/')):
            model_path = os.path.join(exp_dir, 'model.pt')
            if not os.path.exists(model_path):
                continue

            rate, avg_total_reward = evaluate_model(model_path=model_path,env_name='simple_tag_v2')
            cooperation_rates.append(rate)
            cooperation_total_rewards.append(avg_total_reward)

        if cooperation_rates:
            results[algorithm] = {
                'mean': np.mean(cooperation_rates),
                'std': np.std(cooperation_rates),
                'mean total_reward': np.mean(cooperation_total_rewards),
                'std total_reward': np.std(cooperation_total_rewards),
                'samples': len(cooperation_rates)
            }

    # 格式化输出
    print("\nAlgorithm Comparison Results:")
    print("{:<20} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
        'Algorithm', 'Mean', 'Std', 'mean total_reward', 'std total_reward', 'Samples'))
    for algo, data in results.items():
        print("{:<20} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10}".format(
            algo, data['mean'], data['std'], data['mean total_reward'], data['std total_reward'], data['samples']))


if __name__ == '__main__':
    main()