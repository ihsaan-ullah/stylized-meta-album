#!/usr/bin/env python

# Usage: python ingestion.py input_dir output_dir submission_program_dir

# Dependencies
import os
from sys import argv, path

# Configurations
ROOT_DIR = "/home/romain/StylizedMetaAlbum/Archive/"
VERBOSE = True
DEFAULT_INPUT_DIR = ROOT_DIR + "input_data"
DEFAULT_OUTPUT_DIR = ROOT_DIR + "sample_result_submission"
DEFAULT_PROG_DIR = ROOT_DIR + "ingestion_program"
DEFAULT_SUBMISSION_DIR = ROOT_DIR + "sample_code_submission"
DEFAULT_DATA_NAME = "stylized_generation"
VERSION = 1.1

def main(input_dir, output_dir, submission_dir, data_name):
    print(submission_dir)
    # Add path
    #os.chdir(submission_dir)
    path.append(submission_dir)
    try:
        import experiment
        import importlib
        importlib.reload(experiment)
        path.remove(submission_dir)
    except ImportError:
        print(f"experiment not found in {submission_dir}")
        exit(1)       

    experiment.run(output_dir, input_dir)
    
    print(f"\n{'#'*60}\n{'#'*9} Ingestion program finished successfully {'#'*10}\n{'#'*60}")
    path 


if __name__=="__main__":   
    # Get input and output directory names
    if len(argv)==1: # Use the default input and output directories if no arguments are provided
        input_dir = DEFAULT_INPUT_DIR
        output_dir = DEFAULT_OUTPUT_DIR
        submission_dir= DEFAULT_SUBMISSION_DIR
        program_dir = DEFAULT_PROG_DIR
        data_name = DEFAULT_DATA_NAME
    else:
        input_dir = os.path.abspath(argv[1])
        output_dir = os.path.abspath(argv[2])
        program_dir = os.path.abspath(argv[3])
        submission_dir = os.path.abspath(argv[4])
        data_name = DEFAULT_DATA_NAME

    if VERBOSE: 
        print("Using input_dir: " + input_dir)
        print("Using output_dir: " + output_dir)
        print("Using program_dir: " + program_dir)
        print("Using submission_dir: " + submission_dir)
        print("Data name: "+ data_name)
    """submission_dir = "/home/romain/StylizedMetaAlbum/Archive/sample_code_submission/submission_awa"
    subs = os.listdir(submission_dir)
    print(subs)
    for s in subs:
    	os.chdir(f"/home/romain/StylizedMetaAlbum/Archive/ingestion_program")
    	ss = submission_dir + "/" + s
    	print("subs : ", ss)"""
    ss = submission_dir
    main(input_dir, output_dir, ss, data_name)
