'''
    python==3.6.13
    numpy==1.19.5
    torch==1.10.2
    gym==0.26.0
    pygame==2.1.0
    stable-baselines3==1.0
'''


import numpy as np
import gym
import time
from tqdm import tqdm
from gym.envs.toy_text.frozen_lake import generate_random_map
from gym.envs.toy_text import FrozenLakeEnv
import numpy as np



class CustomFrozenLake(FrozenLakeEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def step(self, action):
        state, reward, terminated, truncated, info = super().step(action)

        # 掉进冰洞-1,原版本为0
        if reward == 0 and self.desc.flatten()[state] == b'H': 
            reward = -1
        
        return state, reward, terminated, truncated, info


def train_MC(env, policy, gamma, num_episodes=10000):
    """
    First-visit Monte Carlo
    """
    success_count = 0
    pbar = tqdm(total=num_episodes, desc="Training", dynamic_ncols=True)

    # 初始化 Q(s, a) 和policy
    Q = np.random.randn(env.observation_space.n, env.action_space.n) * 0.1
    returns = {s: {a: [] for a in range(env.action_space.n)} for s in range(env.observation_space.n)} 

    for i in range(num_episodes):
        episode = []  #  (state, action, reward) 
        state, _ = env.reset()
        while True:
            action_probabilities = policy(state, Q)
            action = np.random.choice(env.action_space.n, p=action_probabilities)
            next_state, reward, terminated, truncted, _ = env.step(action)
            if int(reward) == 1:
                success_count += 1
            episode.append((state, action, reward))
            state = next_state
            if terminated or truncted:
                break
        
        G = 0
        visited_states = []

        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = gamma * G + reward 

            # First-visit check
            if (state, action) not in visited_states:
                visited_states.append((state, action))
                returns[state][action].append(G)
                Q[state][action] = np.mean(returns[state][action]) 

        # 可以提前结束
        if success_count >= 10000 and i > 30000:
            break

        pbar.update(1)  # 进度 +1
        pbar.set_postfix(success_count=success_count)  # 更新成功次数

    pbar.close() 
    print("-----Training Done-----\n")  
    print(np.round(Q,3))

    return Q


def train_SARSA(env, policy, gamma, lr, num_episodes=10000):
    """
    First-visit Monte Carlo
    """
    success_count = 0
    pbar = tqdm(total=num_episodes, desc="Training", dynamic_ncols=True)

    # 初始化 Q(s, a) 和policy
    Q = np.random.randn(env.observation_space.n, env.action_space.n) * 0.1

    for i in range(num_episodes):
        episode = []  #  (state, action, reward, next_state) 
        state, _ = env.reset()
        while True:
            action_probabilities = policy(state, Q)
            action = np.random.choice(env.action_space.n, p=action_probabilities)
            next_state, reward, terminated, truncted, _ = env.step(action)
            if int(reward) == 1:
                success_count += 1
            episode.append((state, action, reward, next_state))
            state = next_state
            if terminated or truncted:
                break
        

        for t in range(len(episode)):
            state, action, reward, next_state = episode[t]
            next_action_probabilities = policy(next_state, Q)
            next_action = np.random.choice(env.action_space.n, p=next_action_probabilities)
            # Update
            Q[state][action] += lr * (reward + gamma * Q[next_state][next_action] - Q[state][action])

        # 可以提前结束
        if success_count >= 10000 and i > 20000:
            break

        pbar.update(1)  # 进度 +1
        pbar.set_postfix(success_count=success_count)  # 更新成功次数

    pbar.close() 
    print("-----Training Done-----\n")  
    print(np.round(Q,3))

    return Q


def train_QL(env, policy, gamma, lr, num_episodes=10000):
    """
    First-visit Monte Carlo
    """
    success_count = 0
    pbar = tqdm(total=num_episodes, desc="Training", dynamic_ncols=True)

    # 初始化 Q(s, a) 和policy
    Q = np.random.randn(env.observation_space.n, env.action_space.n) * 0.1

    for i in range(num_episodes):
        episode = []  #  (state, action, reward, next_state) 
        state, _ = env.reset()
        while True:
            action_probabilities = policy(state, Q)
            action = np.random.choice(env.action_space.n, p=action_probabilities)
            next_state, reward, terminated, truncted, _ = env.step(action)
            if int(reward) == 1:
                success_count += 1
            episode.append((state, action, reward, next_state))
            state = next_state
            if terminated or truncted:
                break
        

        for t in range(len(episode)):
            state, action, reward, next_state = episode[t]
            # Update
            Q[state][action] += lr * (reward + gamma * max(Q[next_state]) - Q[state][action])

        # 可以提前结束
        if success_count >= 10000 and i > 20000:
            break

        pbar.update(1)  # 进度 +1
        pbar.set_postfix(success_count=success_count)  # 更新成功次数

    pbar.close() 
    print("-----Training Done-----\n")  
    print(np.round(Q,3))

    return Q


def epsilon_soft_policy(env, epsilon):
    n_actions = env.action_space.n
    def policy(state, Q):
        policy = np.ones(n_actions) * (epsilon / n_actions)  # 初始概率为 ε / |A(s)|
        best_action = np.argmax(Q[state])  # 最优动作
        policy[best_action] = 1 - epsilon + epsilon / n_actions  # 率为 1 - ε + ε / |A(s)|
        return policy
    return policy


def get_env(map_size: int=5, render: bool=False, map=None):
    if map is None:
        map = generate_random_map(size=map_size, p=0.75)
    if render:
        env = CustomFrozenLake(desc=map, is_slippery=False, render_mode='human')
        return env, map
    else:
        env = CustomFrozenLake(desc=map, is_slippery=False) 
        return env, map


def test_policy(env, Q):
    state, _ = env.reset()
    action_type = ['LEFT', 'DOWN', 'RIGHT', 'UP']

    while True:
        action = np.argmax(Q[state])
        print(action_type[action])
        state, reward, terminated, truncted, _ = env.step(action)
        if terminated or truncted:
            env.reset()
    env.close()



if __name__ == "__main__":

    # Settings
    map_size = 20
    episodes = 50000
    epsilon = 0.1
    gamma = 0.98   # discount factor
    lr = 0.1

    # Train
    env, map = get_env(map_size=map_size)
    print('Start Training')
    policy = epsilon_soft_policy(env, epsilon)

    ''' Change Algorithms Here '''
    #Q = train_MC(env, policy, gamma=gamma, num_episodes=episodes)
    #Q = train_SARSA(env, policy, gamma=gamma, lr=lr, num_episodes=episodes)
    Q = train_QL(env, policy, gamma=gamma, lr=lr, num_episodes=episodes)

    env.close()

    # Test
    env_demo,_ = get_env(map=map, render=True)
    test_policy(env_demo, Q)
