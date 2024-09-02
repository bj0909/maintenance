# if torch.cuda.is_available():
#     print("CUDA is available!")
# else:
#     print("CUDA is not available.")
import torch
import numpy as np
import random
import csv
import matplotlib.pyplot as plt
from collections import deque
import torch.nn as nn
import torch.optim as optim
from maintenance_env import RoadPipeMaintenanceEnv
from component_level_models import Road, Pipe, Action
from network_level_models import IntegratedNetwork


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.memory)


def soft_update(local_model, target_model, tau=0.001):
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


def act_bl(state):
    """
    :param state: network state
    :return: baseline actions follow a condition-based maintenance policy - actions are only taken when a component is
    in a failure state.
    """
    actions = []

    # Determine actions for road components
    for i in range(179):  # Assuming 179 roads
        road_state = state[i]
        if road_state in [0, 1, 2, 3]:
            actions.append(0)  # 'Do Nothing'
        elif road_state == 4:
            actions.append(2)  # 'Perfect Maintenance'

    # Determine actions for pipe components
    for i in range(179, 274):  # Assuming 94 pipes follow the 179 roads
        pipe_state = state[i]
        if 40 < pipe_state <= 49:
            actions.append(2)  # 'Perfect Maintenance'
        else:
            actions.append(0)  # 'Do Nothing'

    return np.array(actions)


class DQNAgent:
    # Add the action constraints:
    ROAD_ACTION_MAP = {
        0: [0],  # Only 'Do Nothing' is valid
        1: [0, 1],  # 'Do Nothing' or 'Minor Maintenance'
        4: [2]  # Only 'Perfect Maintenance' is valid
    }

    PIPE_ACTION_MAP = {
        0: [0],  # Only 'Do Nothing' is valid
        10: [0, 1],  # 'Do Nothing' or 'Minor Maintenance'
        49: [2]  # Only 'Perfect Maintenance' is valid
    }

    def __init__(self, state_size, action_size, config):
        self.state_size = state_size
        self.action_size = action_size  # Total number of components (roads + pipes)
        self.num_actions_per_component = 3  # Each component has 3 possible actions: 0, 1, 2
        self.memory = ReplayBuffer(buffer_size=10000, batch_size=64)
        self.gamma = 0.99  # Increase gamma, the agent values future rewards more, o.w. immediate rewards
        self.epsilon = 1.0  # Increase epsilon, the agent explores more initially
        self.epsilon_decay = 0.999  # Increase epsilon_decay, the agent reduces exploration slower
        self.epsilon_min = 0.001  # Increase epsilon_min, the agent maintains a higher exploration rate
        self.learning_rate = 0.0001  # Increase learning_rate, faster learning but risks instability (hard to converge)
        self.update_every = 6  # Increase update_every, less frequent updates, reduce computational load, slow down learning
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.qnetwork_local = QNetwork(state_size, self.action_size).to(self.device)
        self.qnetwork_target = QNetwork(state_size, self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.learning_rate)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        adjusted_actions = action
        self.memory.add(state, adjusted_actions, reward, next_state, done)
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.memory.batch_size:
                experiences = self.memory.sample()
                loss = self.learn(experiences)
                return loss
        return None

    def act(self, state):
        # Convert the state to a PyTorch tensor and move to the appropriate device (CPU or GPU)
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        # Set the network in evaluation mode
        self.qnetwork_local.eval()
        with torch.no_grad():
            # Forward pass to get the action values
            action_values = self.qnetwork_local(state_tensor)
        # Set the network back to training mode
        self.qnetwork_local.train()

        # Flatten the action values for easier manipulation
        action_values = action_values.cpu().data.numpy().flatten()

        # Initialize the actions list
        actions = []

        # Determine actions for road components
        for i in range(179):  # Assuming 179 roads
            road_state = state[i]
            valid_actions = self.ROAD_ACTION_MAP.get(road_state, [0, 1, 2])
            if random.random() > self.epsilon:
                # Exploitation: choose the action with the highest Q-value among valid actions
                valid_action_values = [action_values[i * 3 + a] for a in valid_actions]
                best_action = valid_actions[np.argmax(valid_action_values)]
            else:
                # Exploration: choose a random action from the valid actions
                best_action = random.choice(valid_actions)
            actions.append(best_action)

        # Determine actions for pipe components
        for i in range(179, 274):  # Assuming 94 pipes follow the 179 roads
            pipe_state = state[i]
            valid_actions = self.PIPE_ACTION_MAP.get(pipe_state, [0, 1, 2])
            if random.random() > self.epsilon:
                # Exploitation: choose the action with the highest Q-value among valid actions
                valid_action_values = [action_values[(i - 179) * 3 + a] for a in valid_actions]
                best_action = valid_actions[np.argmax(valid_action_values)]
            else:
                # Exploration: choose a random action from the valid actions
                best_action = random.choice(valid_actions)
            actions.append(best_action)

        # Convert the list of actions to a NumPy array before returning
        return np.array(actions)

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)  # Ensure actions have the correct shape
        rewards = torch.from_numpy(rewards).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones = torch.from_numpy(dones.astype(np.uint8)).float().to(self.device)

        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards.view(-1, 1) + (self.gamma * Q_targets_next * (1 - dones.view(-1, 1)))

        all_q_values = self.qnetwork_local(states)

        actions = actions.view(-1, 1, actions.shape[1])
        all_q_values = all_q_values.view(-1, 1, all_q_values.shape[1])

        Q_expected = all_q_values.gather(2, actions).squeeze(1)
        Q_expected = Q_expected.sum(dim=1).unsqueeze(1)

        loss = nn.MSELoss()(Q_expected, Q_targets)
        print("Loss Value:", loss.item())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        soft_update(self.qnetwork_local, self.qnetwork_target)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()


def save_agent(agent, filename):
    checkpoint = {
        'qnetwork_local': agent.qnetwork_local.state_dict(),
        'qnetwork_target': agent.qnetwork_target.state_dict(),
        'optimizer': agent.optimizer.state_dict(),
        'epsilon': agent.epsilon
    }
    torch.save(checkpoint, filename)
    print(f"Agent saved to {filename}")


def load_agent(agent, filename):
    checkpoint = torch.load(filename)
    agent.qnetwork_local.load_state_dict(checkpoint['qnetwork_local'])
    agent.qnetwork_target.load_state_dict(checkpoint['qnetwork_target'])
    agent.optimizer.load_state_dict(checkpoint['optimizer'])
    agent.epsilon = checkpoint['epsilon']
    print(f"Agent loaded from {filename}")


# R bad
def generate_road_state(partition, seed=None):
    if seed is not None:
        np.random.seed(seed)
    return np.random.choice([3, 4], p=[0.75, 0.25])


# # R medium
# def generate_road_state(partition, seed=None):
#     if seed is not None:
#         np.random.seed(seed)
#     return 2  # Set all road components to an initial state 


# # R good
# def generate_road_state(partition, seed=None):
#     if seed is not None:
#         np.random.seed(seed)
#     return np.random.choice([0, 1], p=[0.5, 0.5])


# P bad
def generate_pipe_state(partition, seed=None):
    if seed is not None:
        np.random.seed(seed)
    if partition in ['sw', 'nw']:
        probabilities = [3] * 10 + [1] * 10  # Adjusted probabilities for more likely and less likely ranges
        normalized_probabilities = np.array(probabilities) / np.sum(probabilities)
        return np.random.choice(np.arange(30, 50), p=normalized_probabilities)
    elif partition in ['se', 'ne']:
        probabilities = [3] * 10 + [1] * 10  # Adjusted probabilities for more likely and less likely ranges
        normalized_probabilities = np.array(probabilities) / np.sum(probabilities)
        return np.random.choice(np.arange(30, 50), p=normalized_probabilities)


# # P medium
# def generate_pipe_state(partition, seed=None):
#     if seed is not None:
#         np.random.seed(seed)
#     if partition in ['sw', 'nw']:
#         # Generating a random integer from 0 to 19 with uniform probabilities
#         return np.random.randint(20, 30)  # np.random.randint is used directly for uniform distribution
#     elif partition in ['se', 'ne']:
#         # Generating a random integer from 35 to 49 with uniform probabilities
#         return np.random.randint(20, 30)


# # P good
# def generate_pipe_state(partition, seed=None):
#     if seed is not None:
#         np.random.seed(seed)
#     if partition in ['sw', 'nw']:
#         # Generating a random integer from 0 to 19 with uniform probabilities
#         return np.random.randint(0, 20)  # np.random.randint is used directly for uniform distribution
#     elif partition in ['se', 'ne']:
#         # Generating a random integer from 35 to 49 with uniform probabilities
#         return np.random.randint(0, 20)


# def apply_random_states(env):
#     road_state_map = {}
#     pipe_state_map = {}

#     # Assuming env.network.roads and env.network.pipes have the attribute 'partition'
#     for road in env.network.roads:
#         if road.partition not in road_state_map:
#             road_state_map[road.partition] = {}
#         road_state_map[road.partition][road.road_id] = generate_road_state(road.partition)

#     for pipe in env.network.pipes:
#         if pipe.partition not in pipe_state_map:
#             pipe_state_map[pipe.partition] = {}
#         pipe_state_map[pipe.partition][pipe.pipe_id] = generate_pipe_state(pipe.partition)

#     env.set_state_by_partition(road_state_map, pipe_state_map)

def apply_random_states(env, seed=None):
    road_state_map = {}
    pipe_state_map = {}

    # Optional: Set the global seed for reproducibility across multiple calls
    if seed is not None:
        np.random.seed(seed)

    # Assuming env.network.roads and env.network.pipes have the attribute 'partition'
    for road in env.network.roads:
        if road.partition not in road_state_map:
            road_state_map[road.partition] = {}
        road_state_map[road.partition][road.road_id] = generate_road_state(road.partition, seed=seed)

    for pipe in env.network.pipes:
        if pipe.partition not in pipe_state_map:
            pipe_state_map[pipe.partition] = {}
        pipe_state_map[pipe.partition][pipe.pipe_id] = generate_pipe_state(pipe.partition, seed=seed)

    env.set_state_by_partition(road_state_map, pipe_state_map)


config = {
    'road_data_file': 'road_ybor_all2.json',
    'pipe_data_file': 'pipe_ybor_25252525_random.json',
    'model_path': 'gradient_boosting_regressor_model_1.pkl',
    'max_steps': 100,
    'failure_threshold': 0.2,
    'use_group_maintenance': True
}

env = RoadPipeMaintenanceEnv(config)

# Flatten the observation space
state_size = np.prod(env.observation_space[0].shape) + np.prod(env.observation_space[1].shape)  # 274: 179 + 94
action_size = sum(space.n for space in env.action_space)  # 822: 274 * 3

# Partitionly set the initial state
# apply_random_states(env)

agent = DQNAgent(state_size, action_size, config)

n_episodes = 1000
max_t = 50  # Number of time steps per episode

# List to store the total rewards for each episode and losses
total_rewards = []
losses = []

for i_episode in range(1, n_episodes + 1):
    state = env.reset()  # Reset the environment at the start of each episode
    state = state.flatten()  # Flatten the state

    total_reward = 0  # To keep track of the total reward per episode
    episode_losses = []  # To keep track of losses for each time step

    for t in range(max_t):
        actions = agent.act(state)  # Use the DQN agent to select actions
        # next_state, reward, done, info, truncated = env.step(actions)
        next_state, reward, done, info, truncated, adjusted_actions = env.step(actions)

        # Manually update the states of roads and pipes in env.network
        next_road_states, next_pipe_states = next_state
        for road, next_road_state in zip(env.network.roads, next_road_states):
            road.state = next_road_state
        for pipe, next_pipe_state in zip(env.network.pipes, next_pipe_states):
            pipe.state = next_pipe_state

        # Concatenate and flatten the next state
        next_state = np.concatenate([next_road_states, next_pipe_states])

        # Update the agent with the experience
        loss = agent.step(state, adjusted_actions, reward, next_state, done)
        if loss is not None:  # Only append valid loss values
            episode_losses.append(loss)

        # Update the total reward
        total_reward += reward
        state = next_state

        # Check if the episode is done
        if done or truncated:
            break

    # Store the total reward for this episode
    total_rewards.append(total_reward)
    # Store the average loss for this episode if there are valid loss values
    if episode_losses:
        losses.append(np.mean(episode_losses))
    else:
        losses.append(0)  # If no valid loss values, append 0 or some other placeholder

    # Output results after each episode
    print(f"Episode {i_episode}: Total Reward = {total_reward}, Steps Taken = {t + 1}, Done = {done}")

# Visualization of losses
plt.figure(figsize=(10, 6))
if len(losses) > 5:
    plt.plot(losses[5:], label='Average Loss per Episode (ignoring first 50 episodes)')
plt.xlabel('Episode')
plt.ylabel('Loss')
plt.title('Loss over Episodes (Ignoring Initial Episodes)')
plt.legend()
plt.grid(True)
plt.show()

# Visualization of rewards (ignoring initial episodes)
plt.figure(figsize=(10, 6))
if len(total_rewards) > 5:
    plt.plot(total_rewards[5:], label='Total Reward per Episode (ignoring first 5 episodes)')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Rewards over Episodes (Ignoring Initial Episodes)')
plt.legend()
plt.grid(True)
plt.show()

# Writing the list to a CSV file
with open('losses.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for item in losses:
        writer.writerow([item])

# Writing the list to a CSV file
with open('rewards.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for item in total_rewards:
        writer.writerow([item])

# Save the trained agent
save_agent(agent, 'dqn_agent_1.pth')

# # Set config. to compare strategies
# # Initialize the environment with group maintenance
# config_group = {
#     'road_data_file': 'road_ybor_all2.json',
#     'pipe_data_file': 'pipe_ybor_25252525_random.json',
#     'model_path': 'gradient_boosting_regressor_model_1.pkl',
#     'max_steps': 100,
#     'failure_threshold': 0.2,
#     'use_group_maintenance': True
# }
# env_group = RoadPipeMaintenanceEnv(config_group)
#
# # Flatten the observation space
# state_size = np.prod(env_group.observation_space[0].shape) + np.prod(
#     env_group.observation_space[1].shape)  # 274: 179 + 94
# action_size = sum(space.n for space in env_group.action_space)  # 822: 274 * 3
#
# # Assuming env is an instance of RoadPipeMaintenanceEnv
# apply_random_states(env_group)  # Don't use random seed!!!!
#
# # Initialize the environment with group maintenance
# config_no_group = {
#     'road_data_file': 'road_ybor_all2.json',
#     'pipe_data_file': 'pipe_ybor_25252525_random.json',
#     'model_path': 'gradient_boosting_regressor_model_1.pkl',
#     'max_steps': 100,
#     'failure_threshold': 0.2,
#     'use_group_maintenance': False
# }
# env_no_group = RoadPipeMaintenanceEnv(config_no_group)
# road_state, pipe_state = env_group.get_state()
# env_no_group.set_state(road_state, pipe_state)
#
# env_bl = RoadPipeMaintenanceEnv(config_no_group)
# env_bl.set_state(road_state, pipe_state)
#
# print(f"Whether the initial state for 3 strategies are same: "
#       f"{env_no_group.get_state() == env_group.get_state() == env_bl.get_state()}")
#
# # To load and use the agent later
# # Initialize the environment and agent
# agent = DQNAgent(state_size=state_size, action_size=action_size, config=config_group)
#
# # Load the trained agent
# load_agent(agent, 'dqn_agent_1.pth')
#
# state = np.concatenate(env_group.get_state()).flatten()
# print(f"initial state: {state}")
#
# with open('rg_pg_proposed.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#
#     headers = [
#         'Timestep', 'Component Type', 'Component ID', 'Action', 'State', 'Total Cost', 'Total Short-term Cost',
#         'Inspection Cost', 'Maintenance Cost', 'Traffic Control Cost', 'Short Term User Cost', 'Total Long-term Cost',
#         'Depreciation Cost', 'Leaking Cost', 'Breakage Cost', 'Long-term Traffic Delay Cost', 'Total Benefit',
#         'Depreciation Benefit', 'Leaking Benefit', 'Breakage Benefit', 'Long Term Traffic Delay Benefit',
#         'Saved Repaving Cost', 'Saved Traffic Control Cost',
#         'Reward'
#     ]
#     writer.writerow(headers)
#
#     # Initial state logging with placeholders for costs and benefits
#     initial_costs_benefits = [0] * 19  # Placeholder values for all cost/benefit columns
#
#     for road in env_group.network.roads:
#         writer.writerow([0, 'Road', road.road_id, 'Initial', road.state] + initial_costs_benefits)
#     for pipe in env_group.network.pipes:
#         writer.writerow([0, 'Pipe', pipe.pipe_id, 'Initial', pipe.state] + initial_costs_benefits)
#
#     total_reward = 0
#     total_cost = 0
#
#     for t in range(50):
#         actions = agent.act(state)
#
#         next_state, reward, done, info, truncated, adjusted_actions = env_group.step(actions)
#
#         # Extract details in the order defined in the header
#         cost_benefit_details = [
#             info['total_cost'],
#             info['total_short_term_cost'],
#             info['inspection_cost'],
#             info['maintenance_cost'],
#             info['traffic_control_cost'],
#             info['short_term_user_cost'],
#             info['total_long_term_cost'],
#             info['depreciation_cost'],
#             info['leaking_cost'],
#             info['breakage_cost'],
#             info['long_term_traffic_delay_cost'],
#             info['total_benefit'],
#             info['depreciation_benefit'],
#             info['leaking_benefit'],
#             info['breakage_benefit'],
#             info['long_term_traffic_delay_benefit'],
#             info['saved_repaving_cost'],
#             info['saved_traffic_control_cost'],
#             reward
#         ]
#
#         # Accumulate total cost
#         total_cost += info['total_cost']
#
#         # Define the labels for the cost and benefit details
#         labels = [
#             'Total Cost', 'Total Short-term Cost', 'Inspection Cost', 'Maintenance Cost',
#             'Traffic Control Cost', 'Short Term User Cost', 'Total Long-term Cost',
#             'Depreciation Cost', 'Leaking Cost', 'Breakage Cost', 'Long-term Traffic Delay Cost', 'Total Benefit',
#             'Depreciation Benefit', 'Leaking Benefit', 'Breakage Benefit', 'Long Term Traffic Delay Benefit',
#             'Saved Repaving Cost', 'Saved Traffic Control Cost', 'Reward'
#         ]
#
#         # Combine the labels with their corresponding values from cost_benefit_details
#         cost_benefit_with_labels = dict(zip(labels, cost_benefit_details))
#
#         # Print the details with labels
#         for label, value in cost_benefit_with_labels.items():
#             print(f"{label}: {value}")
#
#         print(f"Cost and Benefit Details: {cost_benefit_details}")
#
#         # Log the adjusted actions and states for each component
#         for road, action in zip(env_group.network.roads, adjusted_actions[:len(env_group.network.roads)]):
#             writer.writerow([t + 1, 'Road', road.road_id, action, road.state] + cost_benefit_details)
#         for pipe, action in zip(env_group.network.pipes, adjusted_actions[len(env_group.network.roads):]):
#             writer.writerow([t + 1, 'Pipe', pipe.pipe_id, action, pipe.state] + cost_benefit_details)
#
#         # Update state for the next iteration
#         next_road_states, next_pipe_states = next_state
#         next_state = np.concatenate([next_road_states, next_pipe_states])
#         total_reward += reward
#         state = next_state
#
#         # Optional: Break the loop if the environment signals completion
#         if done or truncated:
#             print("Environment has signaled completion.")
#             break
#
#     print(f"Total Reward after loading agent: {total_reward:.6e}")
#     print(f"Total Cost after 5 steps: {total_cost:.6e}")


