from collections import deque
import numpy as np
import random
from Models import PortModel


class PortAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95 #discount rate
        self.epsilon = 1.0 #exporation rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        pm = PortModel(self.state_size, self.action_size)
        self.model = pm.create_model_conv()
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
       
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            x = np.random.rand(self.action_size[0])
            return x/sum(x)
        act_values = self.model.predict(state)
        #best_action = np.argmax(act_values[0])
        return act_values
    
    def replay(self, batch_size=32):
        minibatch = random.sample(self.memory, batch_size)

        states = np.array([tup[0][0] for tup in minibatch])
        actions = np.array([tup[1] for tup in minibatch])
        rewards = np.array([tup[2] for tup in minibatch])
        next_state = np.array([tup[3][0] for tup in minibatch])
        done = np.array([tup[4] for tup in minibatch])

        #Q(s',a)
        target = rewards + self.gamma*self.model.predict(next_state)
        target[done] = rewards[done]
        

        #Q(s,a)
        target_f = self.model.predict(states)
        #make the agent to approximately map the current state to the future disciounted reward
        target_f[range(batch_size), actions] = target

        self.model.fit(states, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def load(self, name):
        self.model.load_weights(name)
    def save(self, name):
        self.model.save_weights(name)
