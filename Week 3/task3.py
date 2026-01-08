import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random

import torch
import torchvision
from torchvision import transforms, datasets

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim


toggle = 0 # toggle = 1 gives results wrt Tabular Q-Learning, else wrt DQN


 # Environment
env = gym.make("MountainCar-v0")

pos_bins = np.linspace(-1.2, 0.6, 20)
vel_bins = np.linspace(-0.07, 0.07, 20)

def discretize_state(state):
    pos, vel = state
    pos_idx = np.digitize(pos, pos_bins)
    vel_idx = np.digitize(vel, vel_bins)
    return pos_idx * (len(vel_bins) + 1) + vel_idx



n_states = (len(pos_bins) + 1) * (len(vel_bins) + 1)
n_actions = env.action_space.n

# -----------------------------
# Q-learning parameters
# -----------------------------
alpha = 0.1        # learning rate
gamma = 0.99       # discount factor

epsilon = 1            # initial exploration rate
epsilon_min = 0.01     # minimum exploration rate
epsilon_decay = 0.995   # exploration decay factor

num_episodes = 10000
max_steps = 200       

# -----------------------------
# Q-table
# -----------------------------
Q = np.zeros((n_states, n_actions))

# -----------------------------
# Tracking performance
# -----------------------------
success_window = deque(maxlen=100)
success_rates = []

# -----------------------------
# Training loop
# -----------------------------

if toggle:
    for episode in range(num_episodes):
        obs, _ = env.reset()
        state = discretize_state(obs)
        success = 0

        for _ in range(max_steps):
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])

            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = discretize_state(next_obs)

            if terminated: # Reached the goal
                Q[state, action] += alpha * (reward - Q[state, action])
                # We need to have 0 for the remaining process, but Q[next_state] maybe non-zero if its a mixed bin, so removed
                success = 1
                break
            else:
                Q[state, action] += alpha * (
                    reward + gamma * np.max(Q[next_state]) - Q[state, action]
                )

            state = next_state

            if truncated:
                success = 0
                break

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        success_window.append(success)
        success_rates.append(np.mean(success_window))

        if episode % 500 == 0:
            print(f"Episode {episode}, Success Rate: {success_rates[-1]:.2f}")

    # -----------------------------
    # Plot learning curve
    # -----------------------------
    plt.figure()
    plt.plot(success_rates)
    plt.xlabel("Episode")
    plt.ylabel("Success Rate (last 100 episodes)")
    plt.title("MountainCar Q-Learning Performance")
    plt.savefig("QL.png")


# -----------------------------------------------
# -----------------------------------------------

# -----------------------------
# DQN
# -----------------------------

class DNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.s = nn.Linear(2, 128)
        self.l1 = nn.Linear(128, 128)
        self.r = nn.Linear(128, 3)

    def forward(self, x):
        x = F.relu(self.s(x))
        x = F.relu(self.l1(x))
        x = self.r(x)
        return x 


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen = capacity)

    def append(self, state, action, reward, next_state, terminated):
        self.buffer.append((state, action, reward, next_state, terminated))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        states = []
        actions = []
        rewards = []
        next_states = []
        terminated = []
        # truncated = [] -> Not used, waste of space
               
        for info in batch:
            s, a, r, n, te = info
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(n)
            terminated.append(te)
            # truncated.append(tr)
        
        return ( # Converting to tensors since pytorch works only with tensors
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions).unsqueeze(1),
            torch.FloatTensor(np.array(rewards, dtype=np.float32)).unsqueeze(1),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(terminated)).unsqueeze(1),
            # torch.FloatTensor(np.array(truncated)).unsqueeze(1)
        )

    def __len__(self):
        return len(self.buffer)
    

class DQN:
    def __init__(self, lr=0.001, gamma=0.99, buffer_size=10000, batch_size=64):
        self.gamma = gamma
        self.batch_size = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Suggested by AI to speedup the training
        
        self.policy_net = DNN().to(self.device)
        self.target_net = DNN().to(self.device)
        
        # Initializing Target weights to match the policy_net
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Since we never train this directly
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.rb = ReplayBuffer(buffer_size)
        self.criterion = nn.MSELoss() # Or nn.SmoothL1Loss()

    def select_action(self, state, epsilon):
        if np.random.random() < epsilon: # Exploration
            return random.choice([0,1,2])
        
        # Exploitation
        with torch.no_grad(): # Faster execution
            # Unsqueezed to convert shape from (2,) to (1,2) -> 1 sample, 2 features
            # The DNN expects a batch of inputs, so need to change the shape
            curr_state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(curr_state)
            return q_values.argmax().item()

    def train_step(self):
        states, actions, rewards, next_states, terminated = self.rb.sample(self.batch_size)
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        terms = terminated.to(self.device)
        # truncs = truncated.to(self.device)
        
        # Computing Current Q(s,a) using policy net
        # .gather(1, actions) picks the Q-value for the specific action we took(starndardly used - AI)
        current_q_values = self.policy_net(states).gather(1, actions)
        
        # Computing target Q using target net
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1) # max returns (value, index)
            
            # If terminated, target is just reward, else as usual(even when truncated)
            target_q_values = rewards + (self.gamma * next_q_values * (1 - terms))
        

        loss = self.criterion(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


if not toggle:
    model = DQN()

    for episode in range(num_episodes):
        state, _ = env.reset()
        success = 0

        for step in range(max_steps):
            action = model.select_action(state, epsilon)

            next_state, reward, terminated, truncated, _ = env.step(action)
            """
            In MountainCar, the reward is -1 for every step until you win. A random
            neural network might take 10,000 steps and never find the top. It just sees
            -1, -1, -1.... It has no gradient to learn from.
            The Solution: You need Reward Shaping. You must give the agent a "hint"
            when it is getting higher.
            """
            # Reward modification for the "Optimal Strategy"
            
            pos = next_state[0]
            modified_reward = reward + 0.5 * abs(pos - (-0.5)) # Encourage moving away from bottom
            
            if pos >= 0.5: # Goal
                modified_reward += 100

            # Updating the DQN
            model.rb.append(state, action, modified_reward, next_state, terminated)

            if len(model.rb) >= model.batch_size and step % 4 == 0: # Change
                model.train_step()
            
            state = next_state

            if terminated:
                success = 1
                break

            if truncated:
                success = 0
                break

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        success_window.append(success)
        success_rates.append(np.mean(success_window))

        model.update_target_net() 

        if (episode+1) % 500 == 0:
            print(f"Episode {episode}, Success Rate: {success_rates[-1]:.2f}")

    # -----------------------------
    # Plot learning curve
    # -----------------------------
    plt.figure()
    plt.plot(success_rates)
    plt.xlabel("Episode")
    plt.ylabel("Success Rate (last 100 episodes)")
    plt.title("MountainCar Q-Learning Performance")
    plt.savefig("DQN.png")
