import streamlit as st  # type: ignore
import pandas as pd
import numpy as np
from stqdm import stqdm # type: ignore
import matplotlib.pyplot as plt

np.random.seed(1)
prices = np.loadtxt('prices_btc_Jan_11_2020_to_May_22_2020.txt', dtype=float)

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

def get_reward(before_btc, btc, before_money, money):
    reward = 0
    if(btc != 0):
        if(before_btc < btc):
            reward = 1
    if(money != 0):
        if(before_money < money):
            reward = 1
            
    return reward

def choose_action(state, eps=0.5):
    if np.random.uniform(0, 1) < eps:
        return np.random.randint(0, 2)
    else:
        return np.argmax(q_table[state])

def take_action(state, action, btc, money):
    return actions[nr_to_actions[action]](prices[state], btc, money)

def act(state, action, theta):
    btc, money = theta
    
    done = False
    new_state = state + 1
    
    before_btc, before_money = btc, money
    btc, money = take_action(state, action, btc, money)
    theta = btc, money
    
    reward = get_reward(before_btc, btc, before_money, money)
    
    if(new_state >= nr_states):
        done = True
    
    return new_state, reward, theta, done

actions = { 'buy' : buy, 'sell': sell, 'wait' : wait}
actions_to_nr = { 'buy' : 0, 'sell' : 1, 'wait' : 2 }
nr_to_actions = { k:v for (k,v) in enumerate(actions_to_nr) }

nr_actions = len(actions_to_nr.keys())
nr_states = len(prices)

q_table = np.random.rand(nr_states, nr_actions)

st.title("Trading Bot using Q Learning")

with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("")
        st.markdown("### Today's price : ***{}***".format(prices[-1]))
        st.markdown("##### Yesterday's price : ***{}***".format(prices[-2]))
        st.markdown("")

        fig1, ax1 = plt.subplots()
        ax1.plot(prices[-10000:])
        # ax1.plot(prices)
        ax1.set_title("Price chart")
        ax1.set_xlabel("Market Days")
        ax1.set_ylabel("Price")
        st.pyplot(fig1)
    
    with col2:
        if st.checkbox("Show Q Table", False):
            st.write(pd.DataFrame(q_table))
        
st.markdown("**Hyperparameters**")

with st.container():
    col1, col2 = st.columns(2)
    with col1:
        btc = st.number_input('Amount of BTC', min_value = 0)
        min_alpha = st.number_input('Minimum Learning Rate', min_value=0.01, max_value=1.0, value=0.02, step=0.01)
        n_episodes = st.slider("No of Episodes", min_value = 3, max_value = 30, value = 5)
        alphas = np.linspace(1.0, min_alpha, n_episodes)
    with col2:
        money = st.number_input('Amount of USD', min_value = 100)
        eps = st.number_input('Exploration Rate', min_value=0.01, max_value=1.0, value=0.5, step=0.01)
        gamma = st.number_input('Discount Factor', min_value=0.01, max_value=1.0, value=1.0, step=0.01)

st.markdown("\nLearning Rate per Episode:")
st.bar_chart(
    pd.DataFrame(
        alphas,
        columns=["Learning Rate"],
        index=[i for i in range(1, n_episodes+1)]
    )
)

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
            for e in stqdm(range(n_episodes), desc="Training Progress"):
            # for e in range(n_episodes):
                total_reward = 0
                state = 0
                done = False
                alpha = alphas[e]
    
                while not done:
                    action = choose_action(state, eps)
                    next_state, reward, theta, done = act(state, action, theta)
                    total_reward += reward
        
                    if(done):
                        rewards[e] = total_reward
                        rewards_list.append(total_reward)
                        st.markdown(f"```Episode {e + 1} of {n_episodes} > total reward : {total_reward}```")
                        break
        
                    q_table[state][action] = q_table[state][action] + alpha * (reward + gamma *  np.max(q_table[next_state]) - q_table[state][action])
                    state = next_state

    with col2:
        fig2, ax2 = plt.subplots()
        ax2.plot(rewards_list)
        ax2.set_title("Reward received per episode:")
        ax2.set_xlabel("Episodes")
        ax2.set_ylabel("Reward")
        st.pyplot(fig2)

    state = 0
    acts = np.zeros(nr_states)
    done = False
    total_reward = 0

    while not done:

        action = choose_action(state)
        next_state, reward, theta, done = act(state, action, theta)
        acts[state] = action
        total_reward += reward
        
        if(done):
            break
        state = next_state

    with st.spinner('Finding Optimal Action'):
        with st.container():
            col1, col2 = st.columns(2)
    
            with col2:
                buys_idx = np.where(acts == 0)
                wait_idx = np.where(acts == 2)
                sell_idx = np.where(acts == 1)

                fig3, ax3 = plt.subplots()
                ax3.plot(buys_idx[0], prices[buys_idx], 'bo', markersize=2, label = "Buy")
                ax3.plot(sell_idx[0], prices[sell_idx], 'ro', markersize=2, label = "Sell")
                ax3.plot(wait_idx[0], prices[wait_idx], 'yo', markersize=2, label = "Hold")
                ax3.set_ylabel("Rewards")
                ax3.set_xlabel("Episodes")
                ax3.set_title("Action action per Episode")
                ax3.legend()
                st.pyplot(fig3)

            with col1:
                action_to_be_taken = acts[-1]
                st.markdown("**Inference**")
                st.markdown("Action to be Taken : ***{}***".format(nr_to_actions[int(action_to_be_taken)]))
                st.markdown("Reward received from experience : ***{}***".format(total_reward))

        st.balloons()