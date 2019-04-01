
import pandas as pd
import numpy as np
from Envs import TradingEnv, SharpeEnv
from Agents import DQAgent
from utils import get_scaler, maybe_make_dir
import matplotlib.pyplot as plt
import time


#csv_dir = "D:/fonden/Data/"
csv_dir = "C:/Users/Phili/Desktop/fond/data/"
eric = pd.read_csv(csv_dir + "ERIC.csv", header =0, sep=";", parse_dates = True, index_col = 0,
names= ["datetime", "bid","ask","open", "high", "low", "close", "avg_price", "volume", "turnover", "trades"])
sand = pd.read_csv(csv_dir + "SAND.csv", header =0, sep=";", parse_dates = True, index_col = 0,
names= ["datetime", "bid","ask","open", "high", "low", "close", "avg_price", "volume", "turnover", "trades"])
eric = eric[eric.index > "2018-01-01"]
sand = sand[sand.index > "2018-01-01"]

ce = eric["close"].values[::-1]
cs = sand["close"].values[::-1]

data = np.array([ce,cs])
train_data = data[:, :200]
test_data = data[:, 200:]

maybe_make_dir("weights")
maybe_make_dir("portfolio_val")
timestamp = time.strftime("%Y%m%d%H%M")
episode = 25
batch_size = 32
runs = {s: [] for s in range(episode)}

def sharpe(vals, period=200):
    vals = np.array(vals)
    rets = np.diff(vals) / vals[:-1]
    return np.sqrt(period)*np.mean(rets)/np.std(rets)
def main():
    env = TradingEnv(train_data)
    state_size = env.observation_space.shape
    action_size = env.action_space.n
    agent = DQAgent(state_size, action_size)
    scaler = get_scaler(env)
    portfolio_value = []
    mode = "train"
    
            
    if mode == "test":
        env = TradingEnv(test_data)
        agent.load(weights)
    for e in range(episode):
        state = env._reset()
        state = scaler.transform([state])
        for time in range(env.n_step):
            action = agent.act(state)
            next_state, reward, done, info = env._step(action)
            next_state = scaler.transform([next_state])
            if mode == "train":
                agent.remember(state, action, reward, next_state,done)
            state = next_state
            runs[e].append(info["cur_val"])
            if done:
                print("Episode: {}/{}, episode end value: {}".format(e+1, episode, info["cur_val"]))
                portfolio_value.append(info["cur_val"])
                break
            if mode == "train" and len(agent.memory) > batch_size:
                agent.replay(batch_size)
    for k, v in runs.items():
        if k % 5 == 0:
            plt.plot(v, label=str(k))
            print("Sharpe hos "+str(k), sharpe(v))
    plt.legend()
    plt.show()
    
main()