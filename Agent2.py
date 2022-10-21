import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import deque, namedtuple
import streamlit as st  # type: ignore
import pandas as pd
from stqdm import stqdm # type: ignore
import matplotlib.pyplot as plt

np.random.seed(1)
prices = np.loadtxt('prices_btc_Jan_11_2020_to_May_22_2020.txt', dtype=float)
prices = prices[-10**5:]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 2**2 # minibatch size

GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4

class ReplayBuffer:
    
    def __init__(self, action_size, buffer_size, batch_size, seed=1):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

    def get_memory(self):
        return self.memory

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed=0):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 8)
        self.fc2 = nn.Linear(8, 6)
        self.fc3 = nn.Linear(6, 4)
        self.fc4 = nn.Linear(4, action_size)
        
    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.sigmoid(x)
        x = self.fc4(x)
        return F.sigmoid(x)

class Agent():
    
    def __init__(self, state_size, action_size, seed=0):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def choose_action(self, state, eps=0.5):
        state = torch.from_numpy(np.array([state])).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def get_reward(self, before_btc, btc, before_money, money):
        reward = 0
        if(btc != 0):
            if(before_btc < btc):
                reward = 1
        if(money != 0):
            if(before_money < money):
                reward = 1
        return reward

    def take_action(self, state, action, btc, money):
        return actions[nr_to_actions[action]](prices[state], btc, money)

    def act(self, state, action, theta):
        btc, money = theta
    
        done = False
        new_state = state + 1
    
        before_btc, before_money = btc, money
        btc, money = self.take_action(state, action, btc, money)
        theta = btc, money
    
        reward = self.get_reward(before_btc, btc, before_money, money)
    
        if(new_state >= nr_states):
            done = True
    
        return new_state, reward, theta, done

    def learn(self, experiences, gamma):
        # Obtain random minibatch of tuples from D
        states, actions, rewards, next_states, dones = experiences

        ## Compute and minimize the loss
        ### Extract next maximum estimated value from target network
        q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        ### Calculate target value from bellman equation
        q_targets = rewards + gamma * q_targets_next * (1 - dones)
        ### Calculate expected value from local network
        q_expected = self.qnetwork_local(states).gather(1, actions)
        
        ### Loss calculation (we used Mean squared error)
        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

agent = Agent(state_size=1, action_size=3, seed=0)

def buy(btc_price, btc, money):
    if(money != 0):
        btc = (1 / btc_price ) * money
        money = 0
    return btc, money

def sell(btc_price, btc, money):
    if(btc != 0):
        money = btc_price * btc
        btc = 0
    return btc, money

def wait(btc_price, btc, money):
    return btc, money

actions = { 'buy' : buy, 'sell': sell, 'wait' : wait}
actions_to_nr = { 'buy' : 0, 'sell' : 1, 'wait' : 2 }
nr_to_actions = { k:v for (k,v) in enumerate(actions_to_nr) }

nr_actions = len(actions_to_nr.keys())
nr_states = len(prices)

st.title("Trading Bot using Q Learning - Deep Q Networks")

with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("")
        st.markdown("### Today's price : ***{}***".format(prices[-1]))
        st.markdown("##### Yesterday's price : ***{}***".format(prices[-2]))
        st.markdown("No of Market days considered : ***{}***".format(len(prices)))
        st.markdown("")

        fig1, ax1 = plt.subplots()
        # ax1.plot(prices[-10000:])
        ax1.plot(prices)
        # ax1.plot(prices)
        ax1.set_title("Price chart")
        ax1.set_xlabel("Market Days")
        ax1.set_ylabel("Price")
        st.pyplot(fig1)

    
    with col2:
        if st.checkbox("Show Q Table", False):
            q_memory = agent.memory.get_memory()
            if len(q_memory) > 0:
                st.write("Q Table currently unavailable")
            else:
                st.write(pd.DataFrame(q_memory))
        
st.markdown("**Hyperparameters**")

with st.container():
    col1, col2 = st.columns(2)
    with col1:
        btc = st.number_input('Amount of BTC', min_value = 0)
        LR = st.number_input('Minimum Learning Rate (Alpha)', min_value=0.01, max_value=1.0, value=0.02, step=0.01)
        n_episodes = st.slider("No of Episodes", min_value = 3, max_value = 30, value = 5)
        alphas = np.linspace(1.0, LR, n_episodes)
    with col2:
        money = st.number_input('Amount of USD', min_value = 100)
        eps = st.number_input('Exploration Rate (Epsilon)', min_value=0.01, max_value=1.0, value=0.5, step=0.01)
        GAMMA = st.number_input('Discount Factor (Gamma)', min_value=0.01, max_value=1.0, value=1.0, step=0.01)

st.markdown("\nLearning Rate per Episode:")
st.bar_chart(
    pd.DataFrame(
        alphas,
        columns=["Learning Rate"],
        index=[i for i in range(1, n_episodes+1)]
    )
)

def dqn(n_episodes=2000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, btc=0, money=100):
    rewards = {}
    rewards_list = []
    scores = []   
    theta = (btc, money)                     # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    
    for i_episode in stqdm(range(1, n_episodes+1), desc="Training Progress", colour="blue"):
        state = 0
        score = 0

        for state in stqdm(range(nr_states), desc="Episode " + str(i_episode)):
            action = agent.choose_action(state, eps)
            next_state, reward, theta, done = agent.act(state, action, theta)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward

            if done:
                rewards[i_episode] = score
                rewards_list.append(score)
                st.markdown(f"```Episode {i_episode} of {n_episodes} > total reward : {score}```")
                break 
        
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        
        # print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
    return rewards, rewards_list

theta = (btc, money) #btc, usd
reward = 0
rewards = {}
rewards_list = []

st.markdown("**Agent Training**")

if st.button("Start Training"):

    with st.container():
        col1, col2 = st.columns(2)

    with col1:
        with st.spinner('Undergoing Training'):
                # agent = Agent(state_size=1, action_size=3, seed=0)
                rewards, rewards_list = dqn(
                            n_episodes = n_episodes,
                            btc = btc,
                            money = money,
                            eps_start = eps,
                            eps_decay = 1
                        )

    with col2:
        fig2, ax2 = plt.subplots()
        ax2.plot(rewards_list)
        ax2.set_title("Reward received per episode:")
        ax2.set_xlabel("Episodes")
        ax2.set_ylabel("Reward")
        st.pyplot(fig2)


    with st.spinner('Finding Optimal Action'):
        state = 0
        theta = (btc, money)
        acts = np.zeros(nr_states)
        total_reward = 0

        while True:
            action = agent.choose_action(state)
            next_state, reward, theta, done = agent.act(state, action, theta)
            acts[state] = action
            total_reward += reward
        
            if(done):
                break
            state = next_state

        with st.container():
            col1, col2 = st.columns(2)
    
            with col2:
                buys_idx = np.where(acts == 0)
                wait_idx = np.where(acts == 2)
                sell_idx = np.where(acts == 1)

                fig3, ax3 = plt.subplots()
                ax3.plot(prices, alpha=0.4, linewidth=1)
                ax3.plot(buys_idx[0], prices[buys_idx], '^', markersize=2, label = "Buy")
                ax3.plot(sell_idx[0], prices[sell_idx], 'v', markersize=2, label = "Sell")
                ax3.plot(wait_idx[0], prices[wait_idx], 'yo', markersize=2, label = "Hold")
                ax3.set_ylabel("Rewards")
                ax3.set_xlabel("Episodes")
                ax3.set_title("Action taken per Episode")
                ax3.legend()
                st.pyplot(fig3)

            with col1:
                action_to_be_taken = acts[nr_states-1]
                st.markdown("**Inference**")
                st.markdown("Action to be Taken : ***{}***".format(nr_to_actions[int(action_to_be_taken)]))
                st.markdown("Reward received from experience : ***{}***".format(total_reward))