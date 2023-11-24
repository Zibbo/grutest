import gymnasium as gym
import grunet
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def add_action_to_state(state, action):
    return torch.cat((state, torch.from_numpy(np.array([action])).float()), dim=0)


env = gym.make('Acrobot-v1')
# Define the neural network architecture
model = grunet.GRUNet(input_size=7, hidden_size=100, output_size=6, num_layers=2)
criterion = nn.MSELoss()
learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Online learning loop
num_episodes = 100  # Define the number of episodes
sequence_length = 10  # Define the length of the sequence
model.train()
i = 0
losses = []
for episode in range(num_episodes):
    random_action = env.action_space.sample()
    # Reset the environment for a new episode
    state = env.reset()[0]
    state = torch.from_numpy(state).float()
    state = add_action_to_state(state, random_action)
    done = False
    
    sequence = [torch.from_numpy(np.zeros(7)).float()]*sequence_length
    sequence.append(state)
    
    
    
    while not done:
        input = torch.stack(sequence[-sequence_length:])
        input = input.unsqueeze(0)
        prediction = model(input)
        random_action = env.action_space.sample()
        next_state, reward, done, truncated, info = env.step(random_action)
        next_state = torch.from_numpy(next_state).float()

        optimizer.zero_grad()
        loss = criterion(prediction.squeeze(0), next_state)
        loss.backward()
        optimizer.step()
        l = loss.item()
        losses.append(l)
        random_action = env.action_space.sample()
        # add the random to next state
        next_state = add_action_to_state(next_state, random_action)
        sequence.append(next_state)
        if i % 1000 == 0:
            print("average loss: ", np.mean(losses[-10000:]))
        i += 1
        