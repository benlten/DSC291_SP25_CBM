import os
import sys
import pandas as pd


def avg(directory):
    data = dict()

    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            dataset, arch = filename.split('_')[:2]
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path, header=None)
            data[f"{dataset}_{arch}"] = df[1].sum() / df[2].sum()

    return data


if __name__ == "__main__":
    print(avg(sys.argv[1]))