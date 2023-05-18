# Noah-Manuel Michael
# Created: 16.05.2023
# Last updated: 16.05.2023
# Get Dutch Wikipedia data
# Data from: 20.04.2023

from datasets import load_dataset

dataset = load_dataset('wikipedia', language='nl', date='20230420', beam_runner='DirectRunner')

for datapoint in dataset:
    print(datapoint)
