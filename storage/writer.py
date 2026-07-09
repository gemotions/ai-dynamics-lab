"""
storage/writer.py

Saves experiment results as JSON.
"""

import json
from pathlib import Path
from datetime import datetime


def save(results: list):
    """
    Save a run to a timestamped JSON file.

    Args:
        results: List of model result dictionaries
    """

    # Create folder name from experiment name
    folder = Path("results")

    # Create timestamped filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    filename = folder / f"{timestamp}.json"

    # Build experiment object
    experiment_data = {
        "version": 1,
        "timestamp": datetime.now().isoformat(),
        "results": results,
    }

    # Write JSON
    with open(filename, "w") as file:
        json.dump(experiment_data, file, indent=4)
        file.write("\n")

    print(f"Saved: {filename}")
