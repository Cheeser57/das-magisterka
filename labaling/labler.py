import pandas as pd
import argparse
import xdas

def load_das(path):
    return xdas.open_dataarray(path)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="log.csv", help="Path to the CSV file containing timestamps")
    parser.add_argument("--das", default="das/recorded.nc", help="path to the DAS data file")
    parser.add_argument("--locations", default="locations.csv", help="Path to the locations file")
    parser.add_argument("--output", default="output", help="Path to the output folder")
    args = parser.parse_args()

    timestaps = pd.read_csv(args.input)
    locations = pd.read_csv(args.locations)
    print((timestaps.head()))
    das_data = load_das(args.das)