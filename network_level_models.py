from component_level_models import Road, Pipe
from utils import load_data
from component_level_models import Action
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def calculate_group_duration(group):
    # Calculate the maximum duration of any single task in the group plus one day
    max_duration = 0
    for item in group:
        if isinstance(item, Road) or isinstance(item, Pipe):
            duration = item.calculate_work_duration()
            if duration > max_duration:
                max_duration = duration
    return max_duration + 1


def merge_groups(groups):
    merged = True
    while merged:
        merged = False
        for i in range(len(groups) - 1):
            for j in range(i + 1, len(groups)):
                if set(groups[i]) & set(groups[j]):
                    groups[i] = list(set(groups[i]) | set(groups[j]))
                    del groups[j]
                    merged = True
                    break
            if merged:
                break
    return groups


class IntegratedNetwork:
    def __init__(self, road_data_file, pipe_data_file):
        self.roads = []  # A list to store Road objects
        self.pipes = []  # A list to store Pipe objects
        self.build_network_from_data(road_data_file, pipe_data_file)

    def build_network_from_data(self, road_data_file, pipe_data_file):
        """Builds the entire network for roads and pipes from data files."""
        road_data = load_data(road_data_file)
        pipe_data = load_data(pipe_data_file)
        self.roads = [Road(**data) for data in road_data]
        self.pipes = [Pipe(**data) for data in pipe_data]

    # def build_network_from_lists(self, roads, pipes):
    #     """Builds the entire network for roads and pipes from given lists."""
    #     self.roads = roads
    #     self.pipes = pipes

    def adjust_actions(self, road_actions, pipe_actions):
        """Adjusts specified actions for roads and pipes with state-based limitations without updating states.

            Args:
                road_actions (list of Action): Actions to be applied to each road.
                pipe_actions (list of Action): Actions to be applied to each pipe.
        """
        # Convert integer actions to Action enums
        road_actions = [Action(action) for action in road_actions]
        pipe_actions = [Action(action) for action in pipe_actions]

        adjusted_road_actions = []
        adjusted_pipe_actions = []

        # Adjust actions for roads
        for road, action in zip(self.roads, road_actions):
            # Enforce state-based action constraints
            if road.state == 0:
                action = Action.DN  # Force 'Do Nothing' if road is in perfect condition
            elif road.state == 4:
                action = Action.PM  # Force 'Preventive Maintenance' if road is in failure state
            elif road.state == 1 and action == Action.PM:
                action = Action.MM  # Prevent 'Preventive Maintenance', allow 'Minor Maintenance' instead if road
                # state is 1

            # Store the adjusted action
            adjusted_road_actions.append(action)

        # Adjust actions for pipes
        for pipe, action in zip(self.pipes, pipe_actions):
            # Enforce state-based action constraints
            if pipe.state == 0:
                action = Action.DN  # Force 'Do Nothing' if pipe is in perfect condition
            elif pipe.state == 49:
                action = Action.PM  # Force 'Preventive Maintenance' if pipe is in failure state
            elif pipe.state <= 10 and action == Action.PM:
                action = Action.MM  # Prevent 'Preventive Maintenance', allow 'Minor Maintenance' instead if pipe
                # state is 10 or less

            # Store the adjusted action
            adjusted_pipe_actions.append(action)

        return adjusted_road_actions, adjusted_pipe_actions

    def apply_actions(self, adjusted_road_actions, adjusted_pipe_actions):
        """Applies adjusted actions to roads and pipes, updating their states.

            Args:
                adjusted_road_actions (list of Action): Adjusted actions to be applied to each road.
                adjusted_pipe_actions (list of Action): Adjusted actions to be applied to each pipe.
        """
        # Apply actions to roads
        for road, action in zip(self.roads, adjusted_road_actions):
            # Apply the action if it's valid
            if action in road.action_space:
                road.action = action
                road.update_state()
            else:
                logging.warning(f"Invalid action '{action}' for Road {road.road_id}.")

        # Apply actions to pipes
        for pipe, action in zip(self.pipes, adjusted_pipe_actions):
            # Apply the action if it's valid
            if action in pipe.action_space:
                pipe.action = action
                pipe.update_state()
            else:
                logging.warning(f"Invalid action '{action}' for Pipe {pipe.pipe_id}.")

    def schedule_maintenance(self, road_actions, pipe_actions):
        """Schedules maintenance based on applied actions (actions are not applied by this function).

        Args:
            road_actions (list of Action): Actions applied to each road.
            pipe_actions (list of Action): Actions applied to each pipe.
        """
        # Then proceed to group components and schedule maintenance
        colocated_groups, individual_roads, individual_pipes = self.group_components(road_actions, pipe_actions)

        schedule = []
        group_maintenance = []

        # Calculate durations for colocated groups
        for group in colocated_groups:
            group_duration = calculate_group_duration(group)
            group_maintenance.append((group, group_duration))

        road_index, pipe_index = 0, 0
        remaining_road_time, remaining_pipe_time = 0, 0

        # Iterate as long as there are roads or pipes to be processed
        while road_index < len(individual_roads) or pipe_index < len(individual_pipes):
            if remaining_road_time == 0 and road_index < len(individual_roads):
                road = individual_roads[road_index]
                remaining_road_time = road.calculate_work_duration()

            if remaining_pipe_time == 0 and pipe_index < len(individual_pipes):
                pipe = individual_pipes[pipe_index]
                remaining_pipe_time = pipe.calculate_work_duration()

            if remaining_road_time > 0 and remaining_pipe_time > 0:
                # Both a road and a pipe are under repair
                min_time = min(remaining_road_time, remaining_pipe_time)
                schedule.append((road, pipe, min_time))
                remaining_road_time -= min_time
                remaining_pipe_time -= min_time
            elif remaining_road_time > 0:
                # Only a road is under repair
                schedule.append((road, None, remaining_road_time))
                remaining_road_time = 0
            elif remaining_pipe_time > 0:
                # Only a pipe is under repair
                schedule.append((None, pipe, remaining_pipe_time))
                remaining_pipe_time = 0

            # Move to the next road or pipe if they are finished
            if remaining_road_time == 0:
                road_index += 1
            if remaining_pipe_time == 0:
                pipe_index += 1

        return {
            'colocated_groups': group_maintenance,
            'schedule': schedule
        }

    def schedule_maintenance_no_group(self, road_actions, pipe_actions):
        """Schedules maintenance for each road and pipe component individually.

        Args:
            road_actions (list of Action): Actions applied to each road.
            pipe_actions (list of Action): Actions applied to each pipe.
        """
        schedule = []
        # Define lists to hold roads and pipes needing repair
        roads_need_repair = [road for road, action in zip(self.roads, road_actions) if action != Action.DN]
        pipes_need_repair = [pipe for pipe, action in zip(self.pipes, pipe_actions) if action != Action.DN]

        road_index, pipe_index = 0, 0
        remaining_road_time, remaining_pipe_time = 0, 0

        # Iterate as long as there are roads or pipes to be processed
        while road_index < len(roads_need_repair) or pipe_index < len(pipes_need_repair):
            if remaining_road_time == 0 and road_index < len(roads_need_repair):
                road = roads_need_repair[road_index]
                remaining_road_time = road.calculate_work_duration()

            if remaining_pipe_time == 0 and pipe_index < len(pipes_need_repair):
                pipe = pipes_need_repair[pipe_index]
                remaining_pipe_time = pipe.calculate_work_duration()

            if remaining_road_time > 0 and remaining_pipe_time > 0:
                # Both a road and a pipe are under repair
                min_time = min(remaining_road_time, remaining_pipe_time)
                schedule.append((road, pipe, min_time))
                remaining_road_time -= min_time
                remaining_pipe_time -= min_time
            elif remaining_road_time > 0:
                # Only a road is under repair
                schedule.append((road, None, remaining_road_time))
                remaining_road_time = 0
            elif remaining_pipe_time > 0:
                # Only a pipe is under repair
                schedule.append((None, pipe, remaining_pipe_time))
                remaining_pipe_time = 0

            # Move to the next road or pipe if they are finished
            if remaining_road_time == 0:
                road_index += 1
            if remaining_pipe_time == 0:
                pipe_index += 1

        return {
            'schedule': schedule
        }

    def group_components(self, road_actions, pipe_actions):
        # Define lists to hold roads and pipes needing repair
        roads_need_repair = [road for road, action in zip(self.roads, road_actions) if action != Action.DN]
        pipes_need_repair = [pipe for pipe, action in zip(self.pipes, pipe_actions) if action != Action.DN]

        temp_groups = []
        individual_roads = []
        individual_pipes = pipes_need_repair[:]  # Initially consider all pipes as individual unless grouped

        for road in roads_need_repair:
            if road.co_lo_pipe:  # Check if there is co-location data available
                colo_pipe_ids = [pipe_info['id'] for pipe_info in road.co_lo_pipe]

                # Finding co-located pipes that also need repair
                colocated_pipes = [pipe for pipe in pipes_need_repair if pipe.pipe_id in colo_pipe_ids]

                if colocated_pipes:
                    # Create or extend a temporary group with this road and its co-located pipes
                    temp_groups.append([road] + colocated_pipes)
                    # Remove these pipes from individual_pipes as they are now grouped
                    individual_pipes = [pipe for pipe in individual_pipes if pipe not in colocated_pipes]
                else:
                    # If there are no colocated pipes that need repair, consider the road individually
                    individual_roads.append(road)
            else:
                # If there is no co-location data, handle this road individually
                individual_roads.append(road)

        # Only merge groups if there are temporary groups formed
        merged_groups = []
        if temp_groups:
            merged_groups = merge_groups(temp_groups)

        # Identify individual components not in any group if there are merged groups
        if merged_groups:
            all_grouped_components = set(item for group in merged_groups for item in group)
            individual_roads = [road for road in individual_roads if road not in all_grouped_components]
            individual_pipes = [pipe for pipe in individual_pipes if pipe not in all_grouped_components]

        return merged_groups, individual_roads, individual_pipes
