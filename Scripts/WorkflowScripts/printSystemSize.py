import os
from DatabankLib.core import initialize_databank
import Workflow_utils

"""
Temporary convenience method to just retrieve the smallest files
"""

def find_smallest_consecutive_files(systems, n):
    """
    Finds the smallest n consecutive systems based on the sum of their TRAJECTORY_SIZE
    and returns their starting index and the corresponding systems.
    """
    if not systems or n <= 0 or n > len(systems):
        print("Invalid input. Ensure systems is not empty and n is within a valid range.")
        return None, []

    # Initialize the smallest sum and starting index
    smallest_sum = float("inf")
    smallest_index = -1
  

    # Iterate through possible consecutive groups of size n
    for i in range(len(systems) - n + 1):
        # Extract the current group
        current_group = systems[i:i+n]

        # Calculate the sum of TRAJECTORY_SIZE for the group
        current_sum = sum(system.get("TRAJECTORY_SIZE", float("inf")) for system in current_group)

        # Update if this group has the smallest sum
        if current_sum < smallest_sum:
            smallest_sum = current_sum
            smallest_index = i
          

    return smallest_index, smallest_sum



def find_smallest_files(systems, n):
    """
    Finds the smallest n systems based on their TRAJECTORY_SIZE and returns their original indexes.
    """
    # Attach the original index to each system for tracking
    indexed_systems = [(i, system) for i, system in enumerate(systems)]
    # Sort systems by TRAJECTORY_SIZE
    sorted_systems = sorted(indexed_systems, key=lambda x: x[1].get("TRAJECTORY_SIZE", float("inf")))
    # Return the smallest n systems and their original indexes
    return sorted_systems[:n]


def get_file_at_index(systems, index):
    """
    Fetches and prints the TRAJECTORY_SIZE for the file at the given index.
    """
    try:
        print(systems[index]["TRAJECTORY_SIZE"])
    except (IndexError, KeyError):
        print(f"Invalid index: {index}. Ensure it's within range and has TRAJECTORY_SIZE.")

def find_closest_consecutive_pair(systems):
    """
    Finds the two consecutive systems with the smallest absolute difference 
    in their TRAJECTORY_SIZE and returns their indexes and the difference.
    """
    if len(systems) < 2:
        print("Not enough systems to find consecutive pairs.")
        return None, None

    smallest_diff = float("inf")
    smallest_pair_indices = (None, None)

    for i in range(len(systems) - 1):
        size1 = systems[i].get("TRAJECTORY_SIZE", float("inf"))
        size2 = systems[i+1].get("TRAJECTORY_SIZE", float("inf"))
        current_diff = abs(size1 - size2)

        if current_diff < smallest_diff:
            smallest_diff = current_diff
            smallest_pair_indices = (i, i + 1)

    return smallest_pair_indices, smallest_diff


def find_consecutive_file_sizes(systems, start,end):
    """
    Calculate total size of files between index start and including end.
    """
    # Extract the current group
    current_group = systems[start:end+1]

    # Calculate the sum of TRAJECTORY_SIZE for the group
    current_sum = sum(system.get("TRAJECTORY_SIZE", float("inf")) for system in current_group)
    return current_sum

if __name__ == "__main__":
  
    # Load systems from Workflow_utils
    systems = Workflow_utils.sorted_databank()

    print(find_consecutive_file_sizes(systems,688,688))
