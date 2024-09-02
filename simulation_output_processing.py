# Parse the text file and extract the required statistics for each simulation.
# Only focus on the "Statistics" part of each simulation output as requested.

import re
import pandas as pd

# Path to the uploaded file
file_path = 'output.txt'

# Regular expression to find the statistics part
regex = re.compile(
    r"Statistics \(avg of \d+\):\n"
    r" RouteLength: (?P<RouteLength>\d+\.\d+)\n"
    r" Speed: (?P<Speed>\d+\.\d+)\n"
    r" Duration: (?P<Duration>\d+\.\d+)\n"
    r" WaitingTime: (?P<WaitingTime>\d+\.\d+)\n"
    r" TimeLoss: (?P<TimeLoss>\d+\.\d+)\n"
    r" DepartDelay: (?P<DepartDelay>\d+\.\d+)\n"
    r" DepartDelayWaiting: (?P<DepartDelayWaiting>\d+\.\d+)"
)

# Data structure to hold the extracted data
data = {
    "RouteLength": [],
    "Speed": [],
    "Duration": [],
    "WaitingTime": [],
    "TimeLoss": [],
    "DepartDelay": [],
    "DepartDelayWaiting": []
}

# Open the file and search for matches
with open(file_path, 'r') as file:
    content = file.read()
    matches = regex.findall(content)
    for match in matches:
        for i, key in enumerate(data.keys()):
            data[key].append(float(match[i]))

# Convert to DataFrame
df = pd.DataFrame(data)

# Saving the dataframe to a CSV file
csv_file_path = "multiple_road_block_test_1_res.csv"
df.to_csv(csv_file_path, index=False)

