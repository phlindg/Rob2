from Models import PortModel
from Agents import PortAgent
from Envs import PortEnv
import pandas as pd
import numpy as np
from prepros import prepare_data
import matplotlib.pyplot as plt
from utils import get_scaler


sand = pd.read_csv("C:/Users/Phili/Desktop/fond/data/SAND.csv", sep=";", header = 0, index_col = 0, parse_dates = True).iloc[::-1]
eric = pd.read_csv("C:/Users/Phili/Desktop/fond/data/ERIC.csv", sep=";", header = 0, index_col = 0, parse_dates = True).iloc[::-1]
sand_ret = sand["Closing price"].pct_change().dropna().values
eric_ret = sand["Closing price"].pct_change().dropna().values

data = np.transpose([sand_ret, eric_ret])
train_data = data[:int(0.9*data.shape[0]), :]
episode=10
batch_size = 32

def main():
    env = PortEnv(train_data, tickers=["SAND", "ERIC"], init_invest=2000,period=20)
    state_size = (data.shape[1], 50, 3)
    action_size = (data.shape[1],1,1)
    agent = PortAgent(state_size, action_size)
    #scaler = get_scaler(env)
    for e in range(episode):
        state = env.reset()
        print(state)
        #state = scaler.transform([state])
        for time in range(env.n_step):
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            #next_state = scaler.transform([next_state])
            agent.remember(state, action, reward, next_state, done)
            print(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("Episode: {}/{}, episode end value: {}".format(e+1, episode, info["cur_val"]))
                print(env.weights)
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

main()