from enum import Enum
import numpy as np
import math


class Action(Enum):
    DN = 0  # Do Nothing
    MM = 1  # Minor Maintenance
    PM = 2  # Perfect Maintenance


class Road:
    road_transition_matrices = {
        Action.DN: np.array([[0.951, 0.049, 0.000, 0.000, 0.000],
                             [0.000, 0.942, 0.055, 0.003, 0.000],
                             [0.000, 0.000, 0.792, 0.188, 0.020],
                             [0.000, 0.000, 0.000, 0.611, 0.389],
                             [0.000, 0.000, 0.000, 0.000, 1.000]]),

        Action.MM: np.array([[0.951, 0.049, 0.000, 0.000, 0.000],
                             [0.951, 0.049, 0.000, 0.000, 0.000],
                             [0.000, 0.942, 0.055, 0.003, 0.000],
                             [0.000, 0.000, 0.792, 0.188, 0.020],
                             [0.000, 0.000, 0.000, 0.611, 0.389]]),

        Action.PM: np.array([[0.951, 0.049, 0.000, 0.000, 0.000],
                             [0.951, 0.049, 0.000, 0.000, 0.000],
                             [0.951, 0.049, 0.000, 0.000, 0.000],
                             [0.951, 0.049, 0.000, 0.000, 0.000],
                             [0.951, 0.049, 0.000, 0.000, 0.000]])
    }
    STATE_SPACE = 5
    # INSPECTION_COST_PER_ROAD = 500  # Original
    INSPECTION_COST_PER_ROAD = 400
    WORK_DURATION_BASE = [1, 5, 9]  # For mm maintenance w.r.t. different road states
    # DAILY_TRAFFIC_CONTROL_COST_PER_METER = 20  # Original
    DAILY_TRAFFIC_CONTROL_COST_PER_METER = 2
    REPAVING_COST_PER_METER = 58
    # MM_MAINTENANCE_COST_PER_METER = 131  # Original value
    MM_MAINTENANCE_COST_PER_METER = 3
    PM_MAINTENANCE_COST_PER_METER = [0, 7, 13, 17, 100]
    # PM_MAINTENANCE_COST_PER_METER = [0, 627, 958, 1336, 1804]  # Original values
    # ANNUAL_DEPRECIATION_COST_PER_VEH_PER_METER = [0.0166, 0.0233, 0.0332, 0.0499, 0.0831]  # Original values
    # ANNUAL_DEPRECIATION_COST_PER_VEH_PER_METER = [0.015, 0.02, 0.03, 0.04, 0.08]
    ANNUAL_DEPRECIATION_COST_PER_VEH_PER_METER = [0.001, 0.003, 0.005, 0.008, 0.015]

    # not for training, only for computing the costs

    def __init__(self, road_id, edge_id, lane_id, num_vehicles, avg_speed, is_boundary, num_colo_pipes, speed_limit,
                 road_length, priority, road_type, num_adj_groups, adj_groups, num_lanes, is_oneway, co_lo_pipe,
                 current_state, traffic_delay_single, partition):
        self.state = current_state
        self.action = Action.DN
        self.action_space = [Action.DN, Action.MM, Action.PM]
        self.road_id = road_id
        self.edge_id = edge_id
        self.lane_id = lane_id
        self.num_vehicles_30min = num_vehicles
        self.avg_speed = avg_speed
        self.is_boundary = is_boundary
        self.num_colo_pipes = num_colo_pipes
        self.speed_limit = speed_limit
        self.road_length = road_length
        self.priority = priority
        self.type = road_type
        self.num_adj_groups = num_adj_groups
        self.adj_groups = adj_groups
        self.num_lanes = num_lanes
        self.is_oneway = is_oneway
        self.co_lo_pipe = co_lo_pipe  # list of dicts
        self.traffic_delay_single = traffic_delay_single
        self.partition = partition

    @property
    def id(self):
        return self.road_id

    def calculate_inspection_cost(self):
        if self.action == Action.DN:
            inspection_cost = 0
        else:
            inspection_cost = Road.INSPECTION_COST_PER_ROAD
        return inspection_cost

    def calculate_work_duration(self):
        if self.action == Action.DN:
            return 0  # No work if 'Doing Nothing'
        base_index = 0 if self.road_length <= 200 else (1 if self.road_length <= 600 else 2)
        base_duration = Road.WORK_DURATION_BASE[base_index]

        if self.state == 4:
            return base_duration + 4  # Extra time for preventive maintenance in failure state
        return base_duration + 4 if self.action == Action.PM else base_duration

    def calculate_traffic_control_cost(self):
        if self.action == Action.DN:
            return 0
        duration = self.calculate_work_duration()
        traffic_control_cost = self.road_length * Road.DAILY_TRAFFIC_CONTROL_COST_PER_METER * duration
        return round(traffic_control_cost, 2)

    # def calculate_maintenance_cost(self):
    #     if self.action == Action.DN:
    #         maintenance_cost = 0
    #     elif self.action == Action.MM:
    #         maintenance_cost = self.road_length * (Road.MM_MAINTENANCE_COST_PER_METER + Road.REPAVING_COST_PER_METER)
    #     else:
    #         maintenance_cost = self.road_length * (Road.PM_MAINTENANCE_COST_PER_METER[self.state] +
    #                                                Road.REPAVING_COST_PER_METER)
    #     return round(maintenance_cost, 2)

    def calculate_repaving_cost(self):
        if self.action == Action.DN:
            return 0
        else:
            repaving_cost = self.road_length * Road.REPAVING_COST_PER_METER
            return round(repaving_cost, 2)

    def calculate_maintenance_cost(self):
        if self.action == Action.DN:
            maintenance_cost = 0
        elif self.action == Action.MM:
            maintenance_cost = self.road_length * Road.MM_MAINTENANCE_COST_PER_METER
        else:  # This assumes the else is for Action.PM
            maintenance_cost = self.road_length * Road.PM_MAINTENANCE_COST_PER_METER[self.state]
        return round(maintenance_cost, 2)

    def calculate_annual_depreciation_cost(self, specified_state=None):
        """
        Calculate annual depreciation cost using the current or specified state.

        Args:
        state (int, optional): The state to use for the calculation. If None, use self.state.

        Returns:
        float: The annual depreciation cost.
        """
        if specified_state is None:
            specified_state = self.state
        annual_depreciation_cost = self.road_length * Road.ANNUAL_DEPRECIATION_COST_PER_VEH_PER_METER[specified_state] * \
                                   self.num_vehicles_30min * 48 * 365 / 1000
        return round(annual_depreciation_cost, 2)

    def update_state(self, random_seed=41):
        # Get the transition matrix for the given action
        transition_matrix = Road.road_transition_matrices[self.action]
        # Update state based on transition probabilities
        np.random.seed(random_seed)
        self.state = np.random.choice(range(self.STATE_SPACE), p=transition_matrix[self.state])
        return self.state


class Pipe:
    STATE_SPACE = 50
    # INSPECTION_COST_PER_PIPE = 500
    INSPECTION_COST_PER_PIPE = 400
    WORK_DURATION_BASE = [1, 3, 5]
    # DAILY_TRAFFIC_CONTROL_COST_PER_METER = 20  # Original value
    DAILY_TRAFFIC_CONTROL_COST_PER_METER = 2
    # REPAVING_COST_PER_METER = 58  # Original
    REPAVING_COST_PER_METER = 100
    # MM_MAINTENANCE_COST_PER_METER = 119  # The original value
    MM_MAINTENANCE_COST_PER_METER = 3
    PM_MAINTENANCE_COST_PER_METER = 50
    # PM_MAINTENANCE_COST_PER_METER = 777  # The original value
    ANNUAL_LEAKING_COST_PER_METER = [0, 3, 6, 12, 1500]
    # ANNUAL_LEAKING_COST_PER_METER = [0, 60, 119, 239, 3883]  # The original values
    UNMET_DEMAND_COST_PER_GAL_PER_DAY = 0.05
    COEFFICIENT_OF_BREAKAGE_RATE_GROWTH = 0.056
    UNIT_PIPE_BREAKAGE_COST = 200  # Original

    def __init__(self, pipe_id, diameter, pipe_length, system_demand_shortfall, water_volume, co_lo_road,
                 num_colo_roads, current_state, partition, failure_rate_0):
        self.state = current_state
        self.action = Action.DN
        self.action_space = [Action.DN, Action.MM, Action.PM]
        self.pipe_id = pipe_id
        self.diameter = diameter
        self.pipe_length = pipe_length
        self.system_demand_shortfall = system_demand_shortfall
        self.water_volume = water_volume  # Assume this has already been converted to a float
        self.co_lo_road = co_lo_road  # This should be a list of dictionaries
        self.num_colo_roads = num_colo_roads
        self.partition = partition
        self.failure_rate_0 = failure_rate_0

    @property
    def id(self):
        return self.pipe_id

    def calculate_inspection_cost(self):
        if self.action == Action.DN:
            inspection_cost = 0
        else:
            inspection_cost = Pipe.INSPECTION_COST_PER_PIPE
        return inspection_cost

    def calculate_work_duration(self):
        if self.action == Action.DN:
            return 0  # No work if 'Doing Nothing'

        base_index = 0 if self.pipe_length <= 150 else (1 if self.pipe_length <= 400 else 2)
        base_duration = Pipe.WORK_DURATION_BASE[base_index]

        if self.state == 49:
            return base_duration + 2  # Extra time for preventive maintenance in failure state
        return base_duration + 2 if self.action == Action.PM else base_duration

    def calculate_traffic_control_cost(self):
        # Check if co_lo_road is None and return 0 immediately if true
        if self.action == Action.DN or self.co_lo_road is None:
            return 0

        # Calculate colo_road_total_length as the sum of "total_length" for each co_lo_road
        colo_road_total_length = sum(road['total_colo_length'] for road in self.co_lo_road)
        duration = self.calculate_work_duration()
        traffic_control_cost = colo_road_total_length * Pipe.DAILY_TRAFFIC_CONTROL_COST_PER_METER * duration
        return round(traffic_control_cost, 2)

    # def calculate_maintenance_cost(self):
    #     if self.action == Action.DN:
    #         maintenance_cost = 0
    #     elif self.action == Action.MM:
    #         maintenance_cost = self.pipe_length * (Pipe.MM_MAINTENANCE_COST_PER_METER + Pipe.REPAVING_COST_PER_METER)
    #     else:
    #         maintenance_cost = self.pipe_length * (Pipe.PM_MAINTENANCE_COST_PER_METER + Pipe.REPAVING_COST_PER_METER)
    #     return round(maintenance_cost, 2)

    def calculate_repaving_cost(self):
        if self.action == Action.DN:
            return 0
        else:
            repaving_cost = self.pipe_length * Pipe.REPAVING_COST_PER_METER
            return round(repaving_cost, 2)

    def calculate_maintenance_cost(self):
        if self.action == Action.DN:
            maintenance_cost = 0
        elif self.action == Action.MM:
            maintenance_cost = self.pipe_length * Pipe.MM_MAINTENANCE_COST_PER_METER
        else:  # This assumes the else is for Action.PM
            maintenance_cost = self.pipe_length * Pipe.PM_MAINTENANCE_COST_PER_METER
        return round(maintenance_cost, 2)

    def calculate_annual_leaking_cost(self, specified_state=None):
        """
        Calculate annual leaking cost using the current or specified state.

        Args:
        state (int, optional): The state to use for the calculation. If None, use self.state.

        Returns:
        float: The annual leaking cost.
        """
        # 5 level of states:
        # level 1: new pipes are in age 0-9
        # level 2: pipes in age 10-19
        # level 3: 20-29
        # level 4: 30-39
        # level 5: 40-49
        if specified_state is None:
            specified_state = self.state
        state_level = specified_state // 10
        annual_leaking_cost = self.pipe_length * Pipe.ANNUAL_LEAKING_COST_PER_METER[state_level]
        return round(annual_leaking_cost, 2)

    def calculate_annual_breakage_cost(self, specified_state=None):
        if specified_state is None:
            specified_state = self.state
        annual_breakage_cost = (
                self.UNIT_PIPE_BREAKAGE_COST * self.pipe_length * self.failure_rate_0 *
                math.exp(self.COEFFICIENT_OF_BREAKAGE_RATE_GROWTH * specified_state)
        )
        return annual_breakage_cost

    def calculate_short_term_user_cost(self):
        daily_transport_volume_gal = self.water_volume
        short_term_user_cost = 0
        if self.action in [Action.MM, Action.PM]:
            short_term_user_cost = daily_transport_volume_gal * Pipe.UNMET_DEMAND_COST_PER_GAL_PER_DAY \
                                   * self.calculate_work_duration()

        return round(short_term_user_cost, 2)

    def update_state(self):
        # Update state based on action
        if self.action == Action.DN:
            self.state = min(self.state + 1, self.STATE_SPACE - 1)  # Ensure state does not exceed the defined maximum
        elif self.action == Action.MM:
            # Assume MM recover the state by 10
            self.state = max(1, self.state - 9)  # Ensure state does not go below 0, then + 1 (degrade to next year)
        elif self.action == Action.PM:
            self.state = 1  # Reset state to 0 for perfect maintenance, then + 1
        return self.state
