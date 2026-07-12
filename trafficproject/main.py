from sumo_rl import SumoEnvironment
from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
import random
import sys

SEED = 42

NUM_SECONDS = 6000 # set end period to this in demandelements.rou.xml
DELTA_TIME = 5
env = SumoEnvironment(
    net_file='networkfile.net.xml',
    route_file='demandelements.rou.xml',
    single_agent=True,
    num_seconds=NUM_SECONDS,
    delta_time=DELTA_TIME,
    sumo_seed=SEED,
)

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim=4, hidden_dim=128):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        y = F.relu(self.fc1(x))
        z = F.relu(self.fc2(y))
        return self.fc3(z)

observation, info = env.reset(seed=SEED)
model = DQN(input_dim=29, output_dim=4)

replay_buffer = []
abcd = NUM_SECONDS // DELTA_TIME

print('Collecting Data')
for step in range(abcd):
    signal = env.traffic_signals["clusterJ4_J5"]
    # with torch.no_grad():
    #     action = model(final_obs).argmax().item()
    action = random.randrange(4)
    observation_new, reward, terminated, truncated, info = env.step(action)
    replay_buffer.append((observation, action, reward, terminated))
    observation = observation_new
    initial_wait = info['system_total_waiting_time'] 
    if truncated:
        break
    

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_train_runs = int(sys.argv[1]) if len(sys.argv) > 1 else len(replay_buffer)

print('Training Model')
for i in range(num_train_runs//2):
    observation, action, reward, done = random.choice(replay_buffer)
    observation = torch.from_numpy(observation)

    q_values = model(observation)
    predicted_q_value = q_values[action]
    target_q = torch.tensor(reward, dtype=torch.float32)

    model.train()
    optimizer.zero_grad()
    loss = F.smooth_l1_loss(predicted_q_value, target_q)
    loss.backward()
    optimizer.step()

print('Greedy Training')
observation, info = env.reset(seed=SEED)
observation = torch.from_numpy(observation)
for i in range(abcd//2):
    epsilon_step = i
    decay = abcd
    progress = (epsilon_step / decay)
    if progress > 1.0:
        progress = 1.0
    epsilon = 1.0 - progress * (1.0 - 0.1)
    if random.random() < epsilon:
        action = random.randrange(4)
    else:
        with torch.no_grad():
            action = model(observation).argmax().item()
    predicted_q_value = model(observation)[action]
    observation_new, reward, terminated, truncated, info = env.step(action)
    target_q = torch.tensor(reward, dtype=torch.float32)
    observation = torch.from_numpy(observation_new)
    model.train()
    optimizer.zero_grad()
    loss = F.smooth_l1_loss(predicted_q_value, target_q)
    loss.backward()
    optimizer.step()
    if truncated:
        break

print('Model Testing')
observation, info = env.reset(seed=SEED)
observation = torch.from_numpy(observation)
for i in range(abcd):
    signal = env.traffic_signals["clusterJ4_J5"]
    with torch.no_grad():
        q_values = model(observation)
        action = q_values.argmax().item()
    observation, reward, terminated, truncated, info = env.step(action)
    observation = torch.from_numpy(observation)
    final_wait = info['system_total_waiting_time']
    if truncated:
        break

print(f"Initial waiting time: {initial_wait}, Final waiting time: {final_wait}")


