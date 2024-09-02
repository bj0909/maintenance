import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Tuple, Discrete
from network_level_models import IntegratedNetwork
from component_level_models import Road, Pipe, Action
from surrogate_model_manager import SurrogateModelManager


def normalize_reward(original_reward):
    log_scaled_reward = np.sign(original_reward) * np.log10(1 + np.abs(original_reward))
    return log_scaled_reward


def standardize_costs(costs):
    mean = np.mean(costs)
    std = np.std(costs)
    if std > 0:  # Avoid division by zero
        return (costs - mean) / std
    else:
        return costs


class RoadPipeMaintenanceEnv(gym.Env):
    # Add the action constraints:
    ROAD_ACTION_MAP = {
        0: [Action.DN],  # Only 'Do Nothing' is valid
        1: [Action.DN, Action.MM],  # 'Do Nothing' or 'Minor Maintenance'
        4: [Action.PM]  # Only 'Perfect Maintenance' is valid
    }

    PIPE_ACTION_MAP = {
        0: [Action.DN],  # Only 'Do Nothing' is valid
        10: [Action.DN, Action.MM],  # 'Do Nothing' or 'Minor Maintenance'
        49: [Action.PM]  # Only 'Perfect Maintenance' is valid
    }

    def __init__(self, config):
        super(RoadPipeMaintenanceEnv, self).__init__()
        self.config = config
        self.network = IntegratedNetwork(config['road_data_file'], config['pipe_data_file'])
        self.surrogate_model_manager = SurrogateModelManager(
            network=self.network,
            model_path=config['model_path'],
            use_group_maintenance=config.get('use_group_maintenance', True)
        )
        self.road_action_spaces = [Discrete(3) for _ in range(179)]  # We have 179 roads
        self.pipe_action_spaces = [Discrete(3) for _ in range(95)]  # We have 94 pipes
        self.action_space = Tuple(self.road_action_spaces + self.pipe_action_spaces)
        self.observation_space = Tuple((
            spaces.MultiDiscrete([Road.STATE_SPACE] * 179),
            spaces.MultiDiscrete([Pipe.STATE_SPACE] * 95)
        ))
        self.max_steps = config.get('max_steps', 100)  # Maximum number of steps per episode
        self.current_step = 0
        self.total_cost = 0
        # self.budget = config.get('budget', float('inf'))  # Budget for maintenance
        self.failure_threshold = config.get('failure_threshold', 0.2)  # Failure rate threshold
        self.reset()

    def get_state(self):
        """
        Get the current state of all roads and pipes in the environment.

        Returns:
            tuple: A tuple containing two arrays of states for roads and pipes respectively.
        """
        road_states = [road.state for road in self.network.roads]
        pipe_states = [pipe.state for pipe in self.network.pipes]
        return road_states, pipe_states

    def set_state(self, road_states, pipe_states):
        """
        Set the states of roads and pipes in the environment.

        Args:
            road_states (list): A list of integers representing the states for each road.
            pipe_states (list): A list of integers representing the states for each pipe.
        """
        if len(road_states) != len(self.network.roads) or len(pipe_states) != len(self.network.pipes):
            raise ValueError(
                "The length of road_states or pipe_states does not match the number of roads or pipes in the network.")

        for road, state in zip(self.network.roads, road_states):
            road.state = state
        for pipe, state in zip(self.network.pipes, pipe_states):
            pipe.state = state

    def set_state_by_partition(self, road_state_map, pipe_state_map):
        """
        Set the states for roads and pipes based on their partition attribute.

        Args:
            env (RoadPipeMaintenanceEnv): The environment object.
            road_state_map (dict): Dictionary mapping partition names to desired states for roads.
            pipe_state_map (dict): Dictionary mapping partition names to desired states for pipes.
        """
        # Update road states
        for road in self.network.roads:
            state = road_state_map.get(road.partition, {}).get(road.road_id)
            if state is not None:
                road.state = state

        # Update pipe states
        for pipe in self.network.pipes:
            state = pipe_state_map.get(pipe.partition, {}).get(pipe.pipe_id)
            if state is not None:
                pipe.state = state

    def adjust_action_space(self):
        self.road_action_spaces = []
        self.pipe_action_spaces = []

        # Adjust the action space for each road
        for road in self.network.roads:
            valid_actions = self.ROAD_ACTION_MAP.get(road.state, [Action.DN, Action.MM, Action.PM])
            self.road_action_spaces.append(Discrete(len(valid_actions)))

        # Adjust the action space for each pipe
        for pipe in self.network.pipes:
            if pipe.state == 0:
                valid_actions = self.PIPE_ACTION_MAP[0]
            elif pipe.state == 49:
                valid_actions = self.PIPE_ACTION_MAP[49]
            elif pipe.state <= 10:
                valid_actions = self.PIPE_ACTION_MAP[10]
            else:
                valid_actions = [Action.DN, Action.MM, Action.PM]
            self.pipe_action_spaces.append(Discrete(len(valid_actions)))

        # Reconstruct the full action space
        self.action_space = Tuple(self.road_action_spaces + self.pipe_action_spaces)

    def reset(self, num_imperfect_roads=120, num_imperfect_pipes=52):
        # Modify this reset method to change the initial state settings
        # Ensure that we have at least one road and one pipe to set to failure state
        if num_imperfect_roads < 1 or num_imperfect_pipes < 1:
            raise ValueError("Number of imperfect roads and pipes must be at least 1 to allocate one failure state.")

        # Set most components to perfect state
        for road in self.network.roads:
            road.state = 0  # Perfect state
        for pipe in self.network.pipes:
            pipe.state = 0  # Perfect state

        # Randomly select some components to be in imperfect state
        imperfect_roads = np.random.choice(self.network.roads, num_imperfect_roads, replace=False)
        imperfect_pipes = np.random.choice(self.network.pipes, num_imperfect_pipes, replace=False)

        # Set exactly one road and one pipe to failure state
        imperfect_roads[0].state = 4  # Set the first randomly selected road to failure state
        imperfect_pipes[0].state = 49  # Set the first randomly selected pipe to failure state

        # Set the remaining roads and pipes to other imperfect states
        for road in imperfect_roads[1:]:
            road.state = np.random.randint(1, Road.STATE_SPACE - 1)  # Imperfect states excluding failure state
        for pipe in imperfect_pipes[1:]:
            pipe.state = np.random.randint(1, Pipe.STATE_SPACE - 1)  # Imperfect states excluding failure state

        self.current_step = 0

        # self.adjust_action_space()
        return self.get_observation()

    def step(self, actions):
        self.current_step += 1

        # Capture the current state before applying actions
        current_road_states = [road.state for road in self.network.roads]
        current_pipe_states = [pipe.state for pipe in self.network.pipes]

        road_actions = actions[:179]
        pipe_actions = actions[179:]

        # Adjust the actions
        adjusted_road_actions, adjusted_pipe_actions = self.network.adjust_actions(road_actions, pipe_actions)

        # Calculate costs based on the current state and planned actions
        total_short_term_cost, inspection_cost, maintenance_cost, traffic_control_cost, short_term_user_cost = \
            self.calculate_short_term_cost(current_road_states, current_pipe_states,
                                           adjusted_road_actions, adjusted_pipe_actions)

        # Apply the adjusted actions to the roads and pipes in the network
        self.network.apply_actions(adjusted_road_actions, adjusted_pipe_actions)

        # Capture the next state after applying actions
        next_road_states = [road.state for road in self.network.roads]
        next_pipe_states = [pipe.state for pipe in self.network.pipes]

        # Calculate costs and benefits based on the current state and planned actions
        total_benefit, depreciation_benefit, leaking_benefit, breakage_benefit, long_term_traffic_delay_benefit, \
        saved_repaving_cost, saved_traffic_control_cost = self.calculate_benefit(current_road_states, next_road_states,
                                                                                 current_pipe_states, next_pipe_states,
                                                                                 road_actions,
                                                                                 pipe_actions)

        total_long_term_cost, depreciation_cost, leaking_cost, breakage_cost, \
        long_term_traffic_delay_cost = self.calculate_long_term_cost(next_road_states, next_pipe_states)

        # reward = normalize_reward(total_benefit - total_short_term_cost)
        reward = normalize_reward(- total_long_term_cost - total_short_term_cost)
        # reward = - total_long_term_cost - total_short_term_cost

        # # Combine all cost components into a single array for standardization
        # all_costs = np.array([depreciation_cost, leaking_cost, breakage_cost, long_term_traffic_delay_cost,
        #                       inspection_cost, maintenance_cost, traffic_control_cost, short_term_user_cost])
        #
        # # Standardize the combined cost components
        # standardized_costs = standardize_costs(all_costs)
        #
        # # Calculate the reward as the negative sum of the standardized costs
        # reward = normalize_reward(-np.sum(standardized_costs))

        done = self.check_if_done(next_road_states, next_pipe_states)

        info = {
            'total_cost': round(total_long_term_cost + total_short_term_cost, 2),
            'total_short_term_cost': round(total_short_term_cost, 2),
            'inspection_cost': round(inspection_cost, 2),
            'maintenance_cost': round(maintenance_cost, 2),
            'traffic_control_cost': round(traffic_control_cost, 2),
            'short_term_user_cost': round(short_term_user_cost, 2),
            'total_long_term_cost': round(total_long_term_cost, 2),
            'depreciation_cost': round(depreciation_cost, 2),
            'leaking_cost': round(leaking_cost, 2),
            'breakage_cost': round(breakage_cost, 2),
            'long_term_traffic_delay_cost': round(long_term_traffic_delay_cost, 2),
            'total_benefit': round(total_benefit, 2),
            'depreciation_benefit': round(depreciation_benefit, 2),
            'leaking_benefit': round(leaking_benefit, 2),
            'breakage_benefit': round(breakage_benefit, 2),
            'long_term_traffic_delay_benefit': round(long_term_traffic_delay_benefit, 2),
            'saved_repaving_cost': round(saved_repaving_cost, 2),
            'saved_traffic_control_cost': round(saved_traffic_control_cost, 2)
        }

        truncated = self.current_step >= self.max_steps

        # Convert adjusted actions back to integers for returning
        adjusted_actions = [action.value for action in adjusted_road_actions + adjusted_pipe_actions]

        return (np.array(next_road_states), np.array(next_pipe_states)), reward, done, info, truncated, adjusted_actions

    def random_action(self, seed=31):
        # Set a seed for reproducibility
        np.random.seed(seed)
        # Initialize all actions to 'DN'
        road_actions = [Action.DN] * len(self.road_action_spaces)
        pipe_actions = [Action.DN] * len(self.pipe_action_spaces)

        # Identify indices of roads and pipes that are failed and must receive PM
        failed_road_indices = [i for i, road in enumerate(self.network.roads) if road.state == 4]
        failed_pipe_indices = [i for i, pipe in enumerate(self.network.pipes) if pipe.state == 49]

        # Set PM action for all failed components
        for idx in failed_road_indices:
            road_actions[idx] = Action.PM
        for idx in failed_pipe_indices:
            pipe_actions[idx] = Action.PM

        # Calculate remaining maintenance slots after allocating to failed components
        remaining_road_slots = max(5 - len(failed_road_indices), 0)
        remaining_pipe_slots = max(5 - len(failed_pipe_indices), 0)

        # Get indices of non-failed roads and pipes that can have MM or PM (not DN-only)
        eligible_road_indices = [i for i, space in enumerate(self.road_action_spaces) if
                                 space.n > 1 and road_actions[i] == Action.DN]
        eligible_pipe_indices = [i for i, space in enumerate(self.pipe_action_spaces) if
                                 space.n > 1 and pipe_actions[i] == Action.DN]

        # Randomly select from eligible roads and pipes to fill remaining maintenance slots
        selected_road_indices = np.random.choice(eligible_road_indices,
                                                 min(len(eligible_road_indices), remaining_road_slots), replace=False)
        selected_pipe_indices = np.random.choice(eligible_pipe_indices,
                                                 min(len(eligible_pipe_indices), remaining_pipe_slots), replace=False)

        # Assign MM or PM randomly to the selected indices
        for idx in selected_road_indices:
            road_actions[idx] = np.random.choice([Action.MM, Action.PM])
        for idx in selected_pipe_indices:
            pipe_actions[idx] = np.random.choice([Action.MM, Action.PM])

        return road_actions + pipe_actions

    def calculate_short_term_cost(self, road_states, pipe_states, road_actions, pipe_actions):
        """Calculate total cost and specific cost components

            Args:
                :param pipe_states: A list including the current states of pipes
                :param road_states: A list including the current states of roads
                :param pipe_actions:
                :param road_actions:
        """
        inspection_cost = 0
        maintenance_cost = 0
        traffic_control_cost = 0

        # Calculate costs for roads
        for idx, (state, action) in enumerate(zip(road_states, road_actions)):
            road = self.network.roads[idx]
            road.action = action  # Set action to calculate costs correctly

            # Costs
            road_inspection_cost = road.calculate_inspection_cost()
            road_maintenance_cost = road.calculate_maintenance_cost()
            road_traffic_control_cost = road.calculate_traffic_control_cost()

            # Update total cost components
            inspection_cost += road_inspection_cost
            maintenance_cost += road_maintenance_cost
            traffic_control_cost += road_traffic_control_cost

        # Calculate short term user cost
        short_term_user_cost = self.surrogate_model_manager.calculate_short_term_traffic_delay_cost(road_actions,
                                                                                                    pipe_actions)

        # Calculate costs for pipes
        for idx, (state, action) in enumerate(zip(pipe_states, pipe_actions)):
            pipe = self.network.pipes[idx]
            pipe.action = action  # Set action to calculate costs correctly

            # Costs
            pipe_inspection_cost = pipe.calculate_inspection_cost()
            pipe_maintenance_cost = pipe.calculate_maintenance_cost()
            pipe_traffic_control_cost = pipe.calculate_traffic_control_cost()
            pipe_short_term_user_cost = pipe.calculate_short_term_user_cost()

            # Update total cost components
            inspection_cost += pipe_inspection_cost
            maintenance_cost += pipe_maintenance_cost
            traffic_control_cost += pipe_traffic_control_cost
            short_term_user_cost += pipe_short_term_user_cost

        total_cost = inspection_cost + maintenance_cost + traffic_control_cost + short_term_user_cost
        return total_cost, inspection_cost, maintenance_cost, traffic_control_cost, short_term_user_cost

    def calculate_benefit(self, current_road_states, next_road_states, current_pipe_states, next_pipe_states,
                          road_actions, pipe_actions):
        total_benefit = 0
        depreciation_benefit = 0
        leaking_benefit = 0
        breakage_benefit = 0
        saved_repaving_cost = 0
        saved_traffic_control_cost = 0

        # Calculate long-term depreciation benefits for roads
        for idx, (current_state, next_state, action) in enumerate(
                zip(current_road_states, next_road_states, road_actions)):
            if action != Action.DN:
                road = self.network.roads[idx]
                current_cost = road.calculate_annual_depreciation_cost(current_state)
                next_cost = road.calculate_annual_depreciation_cost(next_state)
                benefit = (current_cost - next_cost)
                total_benefit += benefit
                depreciation_benefit += benefit

        # Calculate long-term leaking benefits for pipes
        for idx, (current_state, next_state, action) in enumerate(
                zip(current_pipe_states, next_pipe_states, pipe_actions)):
            if action != Action.DN:
                pipe = self.network.pipes[idx]
                current_cost = pipe.calculate_annual_leaking_cost(current_state)
                next_cost = pipe.calculate_annual_leaking_cost(next_state)
                benefit = (current_cost - next_cost)
                total_benefit += benefit
                leaking_benefit += benefit

        # Calculate long-term breakage benefits for pipes
        for idx, (current_state, next_state, action) in enumerate(
                zip(current_pipe_states, next_pipe_states, pipe_actions)):
            if action != Action.DN:
                pipe = self.network.pipes[idx]
                current_cost = pipe.calculate_annual_breakage_cost(current_state)
                next_cost = pipe.calculate_annual_breakage_cost(next_state)
                benefit = (current_cost - next_cost)
                total_benefit += benefit
                breakage_benefit += benefit

        # Calculate long-term traffic delay benefit for roads
        long_term_traffic_delay_benefit = self.surrogate_model_manager.calculate_long_term_traffic_delay_benefit(
            current_road_states, next_road_states)

        # Calculate savings from group maintenance
        if self.surrogate_model_manager.use_group_maintenance:
            saved_repaving_cost, saved_traffic_control_cost = self.calculate_group_maintenance_savings(
                road_actions, pipe_actions)
            total_benefit += (saved_repaving_cost + saved_traffic_control_cost)

        total_benefit += long_term_traffic_delay_benefit
        return round(total_benefit, 2), depreciation_benefit, leaking_benefit, breakage_benefit, \
               long_term_traffic_delay_benefit, saved_repaving_cost, saved_traffic_control_cost

    def calculate_long_term_cost(self, next_road_states, next_pipe_states):
        total_long_term_cost = 0
        depreciation_cost = 0
        leaking_cost = 0
        breakage_cost = 0

        # Calculate long-term depreciation cost for roads
        for idx, next_state in enumerate(next_road_states):
            road = self.network.roads[idx]
            cost = road.calculate_annual_depreciation_cost(next_state)
            depreciation_cost += cost
            total_long_term_cost += cost

        # Calculate long-term leaking cost for pipes
        for idx, next_state in enumerate(next_pipe_states):
            pipe = self.network.pipes[idx]
            cost = pipe.calculate_annual_leaking_cost(next_state)
            leaking_cost += cost
            total_long_term_cost += cost

        # Calculate long-term breakage cost for pipes
        for idx, next_state in enumerate(next_pipe_states):
            pipe = self.network.pipes[idx]
            cost = pipe.calculate_annual_breakage_cost(next_state)
            breakage_cost += cost
            total_long_term_cost += cost

        # Calculate long-term traffic delay benefit for roads
        long_term_traffic_delay_cost = self.surrogate_model_manager.calculate_long_term_traffic_delay_cost(
            next_road_states)

        total_long_term_cost += long_term_traffic_delay_cost
        return round(total_long_term_cost,
                     2), depreciation_cost, leaking_cost, breakage_cost, long_term_traffic_delay_cost

    def calculate_group_maintenance_savings(self, road_actions, pipe_actions):
        saved_repaving_cost = 0
        saved_traffic_control_cost = 0

        # Proceed to group components and schedule maintenance
        colocated_groups, individual_roads, individual_pipes = self.network.group_components(road_actions, pipe_actions)

        # Debug: Check if there are any groups to process
        if not colocated_groups:  # Checks if colocated_groups is empty
            print("No groups to process for savings calculation.")
            return saved_repaving_cost, saved_traffic_control_cost

        for group in colocated_groups:
            saved_traffic_control = 0
            saved_repaving = 0

            for component in group:
                if isinstance(component, Pipe):  # Only process Pipe objects
                    idx = self.network.pipes.index(component)  # Get the index of the pipe
                    action = pipe_actions[idx]  # Get the action assigned to this pipe
                    if action != Action.DN:  # Calculate costs only if maintenance is happening
                        # Calculate and accumulate the costs that could be saved
                        saved_traffic_control += component.calculate_traffic_control_cost()
                        saved_repaving += component.calculate_repaving_cost()

            # Calculate savings assuming grouped maintenance reduces costs by some factor, e.g., 20%
            saved_traffic_control_cost += saved_traffic_control
            saved_repaving_cost += saved_repaving

        return saved_repaving_cost, saved_traffic_control_cost

    def render(self, mode='human', close=False):
        # TODO: Render the environment to the screen or another output
        # TODO: Displaying current state of the environment
        if close:
            return
        print("Displaying current state of the environment")

    def close(self):
        # TODO: Clean up the environment resources
        print("Environment closed")

    def get_observation(self):
        road_states = [road.state for road in self.network.roads]
        pipe_states = [pipe.state for pipe in self.network.pipes]
        # Combine the states into a single array, or return them separately depending on how you want to process them
        return np.concatenate([road_states, pipe_states])

    def check_if_done(self, new_road_states, new_pipe_states):
        if self.current_step >= self.max_steps:
            return True
        # if self.total_cost >= self.budget:
        #     return True
        failed_roads = sum(state == 4 for state in new_road_states)
        failed_pipes = sum(state == 49 for state in new_pipe_states)
        failure_rate = (failed_roads + failed_pipes) / (len(new_road_states) + len(new_pipe_states))
        if failure_rate >= self.failure_threshold:
            return True
        return False
