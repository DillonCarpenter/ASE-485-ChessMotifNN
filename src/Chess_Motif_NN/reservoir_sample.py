import pandas as pd
import random


"""
For creating a random sample of the chess motif dataset using reservoir sampling. 
This allows us to efficiently sample from a large CSV file without loading the entire dataset into memory.
"""
def reservoir_sample(csv_path, sample_size):
    reservoir = []
    count = 0
    for chunk in pd.read_csv(csv_path, usecols=["FEN", "Themes"], chunksize=50000):
        for _, row in chunk.iterrows():
            if len(reservoir) < sample_size:
                reservoir.append(row)
            else:
                # Reservoir replacement rule
                j = random.randint(0, count)
                if j < sample_size:
                    reservoir[j] = row
            count += 1

    return pd.DataFrame(reservoir)

if __name__ == "__main__":
    sample_df = reservoir_sample("data/lichess_puzzle_transformed.csv", 100000)
    print(sample_df.head())
    sample_df.to_csv("data/chess_motif_sample_bigger.csv", index=False)