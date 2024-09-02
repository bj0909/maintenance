import json
import pandas as pd


def load_data(file_path):
    """
        Loads data from a JSON file.

        Parameters:
        - file_path: str, the path to the JSON file to be loaded.

        Returns:
        - data: dict or list, the content of the JSON file.
        """
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def load_actions_from_csv(file_path, id_col, action_col):
    """
    Load maintenance actions from a CSV file.

    Args:
    file_path (str): The path to the CSV file containing the maintenance actions.
    id_col (str): The column name in the CSV that contains the IDs (e.g., 'road_id' or 'pipe_id').
    action_col (str): The column name in the CSV that contains the actions (e.g., 'road_action' or 'pipe_action').

    Returns:
    list of tuples: A list of tuples where each tuple contains an ID and an action.
    """
    try:
        # Load the CSV file
        data = pd.read_csv(file_path)

        # Filter the data to include only rows where the action is either 'MM' or 'PM'
        filtered_data = data[data[action_col].isin(['MM', 'PM'])]

        # Create a list of tuples (id, action)
        actions = list(zip(filtered_data[id_col], filtered_data[action_col]))
        return actions
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


def preprocess_data(data):
    # Preprocess loaded data
    pass
