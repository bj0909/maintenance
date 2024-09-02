import json

# Load the data from the provided JSON file
file_path = 'pipe_ybor_25252525_random.json'
with open(file_path, 'r') as file:
    data = json.load(file)

# Calculate the total length of all pipe segments in meters
total_length = sum(pipe["pipe_length"] for pipe in data)

# Calculate the proportion of each pipe's length to the total length in meters
length_proportions = [pipe["pipe_length"] / total_length for pipe in data]

# Total volume of water usage in the system in gallons per day as provided
total_daily_volume_gal_ybor = 482100

# Calculate the water_volume in gallons for each segment based on its proportion
for i, pipe in enumerate(data):
    pipe["water_volume"] = round(length_proportions[i] * total_daily_volume_gal_ybor, 2)  # Round to 2 decimal places

# Convert the pipe lengths from feet to meters (1 foot = 0.3048 meters)
for entry in data:
    entry['pipe_length'] = round(entry["pipe_length"] * 0.3048, 2)  # Round to 2 decimal places

# Save the modified data back to a new JSON file
new_file_path = 'pipe_ybor_final_25252525_random_updated.json'

with open(new_file_path, 'w') as file:
    json.dump(data, file, indent=4)
