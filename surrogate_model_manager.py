import joblib
import numpy as np
import pandas as pd
from component_level_models import Road

road_type_mapping = {
    'unknown': 0,
    'highway.tertiary': 1,
    'highway.tertiary_link': 2,
    'highway.secondary': 3,
    'highway.primary': 4,
    'highway.secondary_link': 5,
    'highway.primary_link': 6
}


def get_blocked_roads_info(roads_blocked):
    num_roads_blocked = len(roads_blocked)
    avg_num_vehicles = np.mean([road.num_vehicles_30min for road in roads_blocked])
    avg_speed_limit = np.mean([road.speed_limit for road in roads_blocked])
    avg_road_length = np.mean([road.road_length for road in roads_blocked])
    avg_num_adj_groups = np.mean([road.num_adj_groups for road in roads_blocked])
    avg_num_lanes = np.mean([road.num_lanes for road in roads_blocked])
    avg_traffic_delay_single = np.mean([road.traffic_delay_single for road in roads_blocked])
    avg_road_type = np.mean(
        [road_type_mapping.get(road.type, 0) for road in roads_blocked])  # Mapping road types to numbers

    return {
        'num_roads_blocked': num_roads_blocked,
        'avg_num_vehicles': avg_num_vehicles,
        'avg_speed_limit': avg_speed_limit,
        'avg_road_length': avg_road_length,
        'avg_num_adj_groups': avg_num_adj_groups,
        'avg_num_lanes': avg_num_lanes,
        'avg_traffic_delay_single': avg_traffic_delay_single,
        'avg_road_type': avg_road_type
    }


class SurrogateModelManager:
    AVG_AADT = 51600
    # TRAFFIC_DELAY_COST_DOLLAR_PER_HOUR = 38.27
    TRAFFIC_DELAY_COST_DOLLAR_PER_HOUR = 0.01

    def __init__(self, network, model_path, use_group_maintenance=True):
        self.network = network
        self.model = joblib.load(model_path)
        self.use_group_maintenance = use_group_maintenance

        print(f"Total roads in environment: {len(self.network.roads)}")
        print(f"Total pipes in environment: {len(self.network.pipes)}")

    def predict_traffic_delay(self, roads_blocked):
        road_info = get_blocked_roads_info(roads_blocked)
        input_data = pd.DataFrame([road_info])
        traffic_delay_time = self.model.predict(input_data)[0]
        return traffic_delay_time

    def calculate_short_term_traffic_delay_cost(self, road_actions, pipe_actions):
        if self.use_group_maintenance:
            maintenance_schedule = self.network.schedule_maintenance(road_actions, pipe_actions)
        else:
            maintenance_schedule = self.network.schedule_maintenance_no_group(road_actions, pipe_actions)

        short_term_user_cost = 0

        if self.use_group_maintenance:
            for group, days in maintenance_schedule['colocated_groups']:
                roads_to_repair = [component for component in group if isinstance(component, Road)]
                traffic_delay_time = max(0, self.predict_traffic_delay(roads_to_repair))
                traffic_delay_cost = traffic_delay_time * self.AVG_AADT * self.TRAFFIC_DELAY_COST_DOLLAR_PER_HOUR / 3600 * days
                short_term_user_cost += traffic_delay_cost

        for road, pipe, days in maintenance_schedule['schedule']:
            if road:
                roads_to_repair = [road]
                traffic_delay_time = max(0, self.predict_traffic_delay(roads_to_repair))
                traffic_delay_cost = traffic_delay_time * self.AVG_AADT * self.TRAFFIC_DELAY_COST_DOLLAR_PER_HOUR / 3600 * days
                short_term_user_cost += traffic_delay_cost

        return round(short_term_user_cost, 2)

    def calculate_long_term_traffic_delay_benefit(self, current_road_states, next_road_states):
        speed_factors = {0: 1.0, 1: 0.96, 2: 0.88, 3: 0.75, 4: 0.54}  # Original
        # speed_factors = {0: 1.0, 1: 0.96, 2: 0.88, 3: 0.82, 4: 0.54}
        total_benefit = 0

        for road, current_state, next_state in zip(self.network.roads, current_road_states, next_road_states):
            if current_state != next_state:  # Only consider roads with state changes
                current_speed = road.avg_speed * speed_factors[current_state]
                next_speed = road.avg_speed * speed_factors[next_state]

                #  Benefit could be negative because if DN, next state could be worse
                benefit = (
                        (road.road_length / current_speed - road.road_length / next_speed)
                        * road.num_vehicles_30min * 48 / 60
                        * self.TRAFFIC_DELAY_COST_DOLLAR_PER_HOUR * 365
                )
                total_benefit += benefit

        return round(total_benefit, 2)

    def calculate_long_term_traffic_delay_cost(self, next_road_states):
        # speed_factors = {0: 1.0, 1: 0.96, 2: 0.88, 3: 0.75, 4: 0.54}
        speed_factors = {0: 1.0, 1: 0.96, 2: 0.88, 3: 0.82, 4: 0.54}
        total_long_term_traffic_delay_cost = 0

        for road, next_state in zip(self.network.roads, next_road_states):
            next_speed = road.avg_speed * speed_factors[next_state]

            #  Benefit could be negative because if DN, next state could be worse
            cost = (
                    (road.road_length / next_speed)
                    * road.num_vehicles_30min * 48 / 60
                    * self.TRAFFIC_DELAY_COST_DOLLAR_PER_HOUR * 365
            )
            total_long_term_traffic_delay_cost += cost

        return round(total_long_term_traffic_delay_cost, 2)
