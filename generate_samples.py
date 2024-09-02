import random
import os
import logging
from simulation_manager import SimulationManager, update_network_file


def generate_random_road_sets(roads, num_samples, num_roads_to_block):
    """
    Generate unique random sets of roads to block.

    Parameters:
    - roads: list of Road objects, the available roads.
    - num_samples: int, number of random samples to generate.
    - max_roads_to_block: int, maximum number of roads to block in each sample.

    Returns:
    - List of lists, where each inner list contains Road objects to be blocked.
    """
    road_sets = set()
    while len(road_sets) < num_samples:
        roads_to_block = tuple(sorted(random.sample(roads, num_roads_to_block), key=lambda road: road.road_id))
        road_sets.add(roads_to_block)
    return [list(road_set) for road_set in road_sets]


def main():
    logging.basicConfig(level=logging.INFO)

    num_samples = 1  # Number of samples to generate
    num_roads_to_block = 3  # Number of roads to block in each sample
    road_data_file = 'road_ybor_all2.json'  # Path to the road data JSON file
    pipe_data_file = 'pipe_ybor_25252525_random.json'  # Path to the pipe data JSON file

    # Initialize the SimulationManager
    simulation_manager = SimulationManager(road_data_file=road_data_file, pipe_data_file=pipe_data_file)

    # Generate random sets of roads to block
    road_sets = generate_random_road_sets(simulation_manager.network.roads, num_samples, num_roads_to_block)

    # File to save the results
    results_file = 'simulation_results_3roads.csv'
    with open(results_file, 'w') as f:
        f.write('blocked_road_ids,traffic_delay\n')

        # Run simulations for each set of blocked roads
        for roads_to_block in road_sets:
            try:
                # Update network file with the blocked roads
                updated_network_file_path = update_network_file(roads_to_block, simulation_manager.NET_FILE_PATH,
                                                                simulation_manager.OUTPUT_NET_FILE_PATH)

                # Run SUMO simulation and get traffic delay
                traffic_delay = simulation_manager.run_traffic_simulation()

                # Record the results
                blocked_road_ids = ','.join(road.road_id for road in roads_to_block)
                f.write(f'"{blocked_road_ids}",{traffic_delay}\n')

                # Clean up temporary files
                os.remove(updated_network_file_path)
                # os.remove(updated_config_file_path)

            except Exception as e:
                print(f"Error during simulation: {e}")


if __name__ == "__main__":
    main()
