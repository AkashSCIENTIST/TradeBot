from matplotlib.pyplot import ylabel
from functions import *
import streamlit as st  # type: ignore
import pandas as pd
import numpy as np
import pydeck as pdk # type: ignore
import plotly.express as px # type: ignore

st.title("Trading Bot using Q Learning")

with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Today's price : ***{}***".format(prices[-1]))
        st.markdown("##### Yesterday's price : ***{}***".format(prices[-2]))
        if st.checkbox("Show Q Table", False):
            st.write(pd.DataFrame(q_table))
    with col2:
        fig1, ax1 = plt.subplots()
        ax1.plot(prices[-10000:])
        # ax1.plot(prices)
        ax1.set_title("Price chart")
        ax1.set_xlabel("Market Days")
        ax1.set_ylabel("Price")
        st.pyplot(fig1)

st.markdown("**Hyperparameters**")

with st.container():
    col1, col2 = st.columns(2)
    with col1:
        btc = st.number_input('Amount of BTC', min_value = 0)
        min_alpha = st.number_input('Minimum Learning Rate', min_value=0.01, max_value=1.0, value=0.02, step=0.01)
        n_episodes = st.slider("No of Episodes", min_value = 5, max_value = 30, value = 5)
        alphas = np.linspace(1.0, min_alpha, n_episodes)
    with col2:
        money = st.number_input('Amount of USD', min_value = 100)
        eps = st.number_input('Exploration Rate', min_value=0.01, max_value=1.0, value=0.5, step=0.01)
        gamma = st.number_input('Discount Factor', min_value=0.01, max_value=1.0, value=1.0, step=0.01)

st.markdown("\nLearning Rate per Episode:")
st.bar_chart(
    pd.DataFrame(
        alphas,
        columns=["Learning Rate"]
    )
)

theta = (btc, money) #btc, usd
reward = 0
rewards = {}
rewards_list = []

st.markdown("**Model Training**")
if st.button("Start Training"):
    st.markdown("Started Training ...")

    with st.container():
        col1, col2 = st.columns(2)

    with col1:
        for e in range(n_episodes):
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
        ax1.set_title("Reward received per episode:")
        ax1.set_xlabel("Episodes")
        ax1.set_ylabel("Reward")
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

    action_to_be_taken = acts[-1]
    st.markdown("**Inference**")
    st.markdown("Action to be Taken : ***{}***".format(nr_to_actions[int(action_to_be_taken)]))
    st.markdown("Reward received from experience : ***{}***".format(total_reward))