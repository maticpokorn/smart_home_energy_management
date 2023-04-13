import pandas as pd
from tqdm import tqdm

import DQN
import DQN_env
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm

EPSILON_INIT = 1
MEMORY_CAPACITY = 2000
epsilon = EPSILON_INIT

seed = 42
n_days = 2
n_steps = 7 * 24 * 4
df = pd.read_csv('working_data.csv')
battery_size = 100

#avg_amount_paid = np.mean(np.multiply(np.array(df['Energy_Consumption']), np.array(df['SMP'])))
median_market_price = np.median(np.array(df['SMP']))


def run_episode(n_steps, env1, env2, dqn, epsilon, a, b):
    seed = np.random.randint(0, 1000)
    # ------------------------------------------------------
    env1.reset(seed)
    state = env1.next_observation_normalized()
    cummulative_rewardRL = []
    actions = []
    battery_too_full = []
    rewards = []
    sRL = 0
    # ------------------------------------------------------
    env2.reset(seed)
    cummulative_reward_rand = []
    s_rand = 0
    for step in range(n_steps):

        action_rand = 1  # np.random.randint(0, 2)
        obs_rand, reward_rand, terminated_rand = env2.step(action_rand)
        # -------------------------------------------
        action = dqn.choose_action(state, epsilon)
        actions.append(action)
        obs, reward, terminated = env1.step(action)
        past_state = state
        state = env1.next_observation_normalized()

        capacity = obs[1, -1]
        past_capacity = obs[1, -2]
        market_price = obs[5, -1]

        my_reward = battery_penalty_expand(capacity, env1.full_battery_capacity, 0.2, 0.7) \
                    + 10 * a * slope_market_price(capacity, past_capacity, market_price, median_market_price) \
                    + b * reward \
                    + 0 * action_price(action, market_price, median_market_price)

        rewards.append(my_reward)
        battery_too_full.append(1 if obs[1, -1] > env1.full_battery_capacity * 0.8 else 0)

        # the reinforcement learning tends to save some energy in the battery, therefore we
        # "sell" all the energy left in the battery and add it to the cumulative reward
        if step == n_steps - 1:
            left_in_battery = obs[1, -1]
            last_price = obs[5, -1]
            left_in_battery_sold = left_in_battery * last_price
            sRL += left_in_battery_sold

            left_in_battery_rand = obs_rand[1, -1]
            last_price_rand = obs_rand[5, -1]
            left_in_battery_sold_rand = left_in_battery_rand * last_price_rand
            s_rand += left_in_battery_sold_rand

        s_rand += reward_rand
        cummulative_reward_rand.append(s_rand)

        sRL += reward
        cummulative_rewardRL.append(sRL)

        dqn.store_transition(past_state, action, my_reward, state)

        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()

        if terminated:
            break

    if env1.test:
        history = np.array(env1.history)
        steps = history[:, 0]
        battery_capacity = history[:, 1]
        energy_consumption = history[:, 3]
        market_price = history[:, 5] * 100
        amount_paid = history[:, 6] * 100
        time_of_day = history[:, 7]
        # print(np.mean(energy_consumption))
        fig, ax = plt.subplots()
        ax.plot(steps, battery_capacity, label='DQN battery charge', c='blue', linewidth=0.7)
        # plt.plot(steps, amount_paid, label='amount paid', c='brown', linewidth=0.5)
        ax.plot(steps, market_price, label='market price', c='orange', linewidth=0.5)
        ax.plot(steps, np.array(cummulative_rewardRL) / 5, label='DQN cost', c='darkblue', linewidth=0.7)
        ax.plot(steps, np.array(actions) * 10, label='action', c='lightgreen', linewidth=0.5, alpha=0.4)
        ax.plot(steps, np.array(rewards) * 10, c='red', linewidth=0.3, alpha=0.5)
        '''
        cmap = cm.get_cmap('Purples')
        for i in range(n_steps - 1):
            h = ax.get_ylim()[1] - ax.get_ylim()[0]
            w = 1
            c1 = (steps[i], plt.ylim()[0])
            rect = patches.Rectangle(c1, w, h, color=cmap(time_of_day[i]))
            ax.add_patch(rect)
        '''
        history_rand = np.array(env2.history)
        ax.plot(steps, np.array(cummulative_reward_rand) / 5, label='compare cost', c='magenta', linewidth=0.7)
        battery_capacity_rand = history_rand[:, 1]
        ax.plot(steps, battery_capacity_rand, label='compare battery charge', c='magenta', linewidth=0.7, alpha=0.5)

        ax.legend(loc='lower left', prop={'size': 6})
        fig.show()

    return sRL, s_rand


def print_episode_results(s1, s2, episode, epsilon):
    print("Episode:", episode, "| Epsilon: ", round(epsilon, 2), "| DQN cost:", round(s1, 2),
          "| compare cost:",
          round(s2, 2), "| Gain over compare:", round(s1 - s2, 2), "(",
          round(100 * (s1 - s2) / abs(s1), 2), "%)")


def action_price(action, market_price, median_market_price):
    above_median = 1 if market_price > median_market_price else 0
    if (market_price > median_market_price) == (action == 1):
        return 1
    return -1


def battery_penalty(capacity, full_capacity):
    relative_c = capacity / full_capacity
    if relative_c < 0.5:
        f = - ((4 * relative_c - 2) ** 2) * 2
    else:
        f = - ((4 * relative_c - 2) ** 2) * 4
    return f + 2


def battery_penalty_expand(capacity, full_capacity, zero_low, zero_high):
    x = capacity / full_capacity
    if x < zero_low:
        f = - (2 / zero_low * x - 2) ** 2 / 2
    elif x > zero_high:
        f = - (2 / (1 - zero_high) * (x - zero_high)) ** 2 * 4
    else:
        f = 0
    return f + 1


def slope_market_price(capacity, past_capacity, market_price, avg_market_price):
    slope = capacity - past_capacity
    relative_market_price = market_price - avg_market_price
    return - (slope * relative_market_price)



def simulate(a, b):

    envRL = DQN_env.Env(df, battery_size, n_days, n_steps)
    envRL.reset(seed)
    dqn = DQN.DQN(envRL.next_observation_normalized().shape[0], 2)

    env_rand = DQN_env.Env(df, battery_size, n_days, n_steps)
    env_rand.reset(seed)

    n_episodes = 300
    epsilon = 1
    for episode in tqdm(range(n_episodes)):
        epsilon = epsilon * 0.98
        cost_dqn, cost_comp = run_episode(n_steps, envRL, env_rand, dqn, epsilon, a, b)
        #print_episode_results(cost_dqn, cost_comp, str(episode) + "/" + str(n_episodes), epsilon)

    # TESTING
    print("--- TESTING ---")
    test_n_steps = len(df) - 30000
    envRL_test = DQN_env.Env(df, full_battery_capacity=battery_size, n_days=n_days, n_steps=test_n_steps, low=30000, high=len(df), test=True)
    env_comp_test = DQN_env.Env(df, full_battery_capacity=battery_size, n_days=n_days, n_steps=test_n_steps, low=30000, high=len(df), test=True)

    epsilon = 0.01
    # ------------------------------------------------------
    cost_dqn, cost_comp = run_episode(test_n_steps, envRL_test, env_comp_test, dqn, epsilon, a, b)
    print_episode_results(cost_dqn, cost_comp, "Testing episode", epsilon)

    return cost_dqn


simulate(3.5, 1.5)
'''
da = {}
for a in range(3):
    db = {}
    for b in range(3):
        db[b] = simulate(a, b)
    da[a] = db

print(da)
'''

