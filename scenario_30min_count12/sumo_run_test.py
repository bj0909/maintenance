import traci
import os


def simulate_traffic(config_file_path):
    sumo_cmd = ["sumo", "-c", config_file_path]
    traci.start(sumo_cmd)

    total_duration = 0
    total_vehicles = 0
    entry_times = {}
    exit_times = {}

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()

        # Capture vehicle entry and exit times
        for veh_id in traci.vehicle.getIDList():
            if veh_id not in entry_times:
                entry_times[veh_id] = traci.simulation.getTime()
            exit_times[veh_id] = traci.simulation.getTime()

    traci.close()

    # Calculate total duration and average duration
    for veh_id in entry_times:
        total_duration += (exit_times[veh_id] - entry_times[veh_id])
        total_vehicles += 1

    avg_duration = total_duration / total_vehicles if total_vehicles > 0 else 0

    print("Total Vehicles:", total_vehicles)
    print("Average Duration:", avg_duration)

    return avg_duration


def main():
    config_path = "osm.sumocfg"
    avg_duration = simulate_traffic(config_path)
    print("Traffic delay (average duration):", avg_duration)


if __name__ == "__main__":
    main()
