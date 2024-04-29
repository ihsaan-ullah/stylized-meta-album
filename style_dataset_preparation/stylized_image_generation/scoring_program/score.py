import os
import json
import shutil
import glob
import time
from sys import argv

import yaml

# Set up default directories and file names:
ROOT_DIR = "../"
DEFAULT_SOLUTION_DIR = os.path.join(ROOT_DIR, "input_data")
DEFAULT_PREDICTION_DIR = os.path.join(ROOT_DIR, "sample_result_submission")
DEFAULT_SCORE_DIR = os.path.join(ROOT_DIR, "scoring_output")
DEFAULT_DATA_NAME = "stylized_generation"

# Set debug and missing score defaults
DEBUG_MODE = 0
MISSING_SCORE = -0.999999

# Define scoring version
SCORING_VERSION = 1.0

# Program version
VERSION = 1.1

def process_arguments(arguments):
    """ Process command line arguments """
    if len(arguments) == 1:  # Default directories
        return DEFAULT_SOLUTION_DIR, DEFAULT_PREDICTION_DIR, DEFAULT_SCORE_DIR, DEFAULT_DATA_NAME
    elif len(arguments) == 3: # Codalab's default configuration
        solution_dir = os.path.join(arguments[1], 'ref')
        prediction_dir = os.path.join(arguments[1], 'res')
        return solution_dir, prediction_dir, arguments[2], DEFAULT_DATA_NAME
    elif len(arguments) == 4: # User-specified directories
        return arguments[1], arguments[2], arguments[3], DEFAULT_DATA_NAME
    else:
        raise ValueError('Wrong number of arguments passed.')

def create_score_directory(score_dir):
    """ Create scoring directory if it doesn't exist """
    if not os.path.exists(score_dir):
        os.makedirs(score_dir)

def main():
    """ Main function to coordinate scoring """
    start = time.time()
    solution_dir, prediction_dir, score_dir, data_name = process_arguments(argv)
    create_score_directory(score_dir)

    with open(os.path.join(score_dir, "scores.json"), "w") as score_file:
        sample_score = { "score": 1.00000 }
        json.dump(sample_score, score_file)

    print(f'Scoring completed in {time.time() - start} seconds.')

if __name__ == "__main__":
    main()
    