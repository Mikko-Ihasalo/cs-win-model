import sys
import os

# Get the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.utils.demo_preprocessing import (
    parse_multiple_demos,
    save_parsed_data,
)
import os


DEMO_FOLDER_PATH = os.path.join(project_root, "data/demos/")
OUTPUT_FOLDER_PATH = os.path.join(project_root, "data/data/")

df = parse_multiple_demos(DEMO_FOLDER_PATH)

save_parsed_data(df, OUTPUT_FOLDER_PATH + "processed_demo.csv")
