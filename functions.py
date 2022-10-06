#import plotly.graph_objects as go #type  :ignore
import matplotlib.pyplot as plt
import numpy as np
import os

prices = np.loadtxt('prices_btc_Jan_11_2020_to_May_22_2020.txt', dtype=float)
# print('Number of prices:', len(prices))

#graph_fig = go.Figure(data=go.Scatter(y=prices[-10000:]))#, x="Date", y="Price", title="Date vs Price")
# fig.show()

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
    # do nothing
    return btc, money

np.random.seed(1)

# action space
actions = { 'buy' : buy, 'sell': sell, 'wait' : wait}
actions_to_nr = { 'buy' : 0, 'sell' : 1, 'wait' : 2 }
nr_to_actions = { k:v for (k,v) in enumerate(actions_to_nr) }

nr_actions = len(actions_to_nr.keys())
nr_states = len(prices)

# q-table = reference table for our agent to select the best action based on the q-value
q_table = np.random.rand(nr_states, nr_actions)
# print("Q Table at Start:")
# print(q_table)

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

