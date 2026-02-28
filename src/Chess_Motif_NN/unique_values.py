import pandas as pd
import sys

def main():
    if len(sys.argv) != 3:
        print("Usage: python unique_motifs.py <csv_file> <column_name>")
        sys.exit(1)

    csv_file = sys.argv[1]
    column = sys.argv[2]

    df = pd.read_csv(csv_file)

    if column not in df.columns:
        print(f"Column '{column}' not found in CSV.")
        print("Available columns:")
        print(df.columns.tolist())
        sys.exit(1)

    unique = set()

    for entry in df[column].dropna():
        # split on whitespace, handle multiple motifs
        parts = entry.split()
        for p in parts:
            unique.add(p)

    unique = sorted(unique)

    print("{")
    for val in unique:
        print(f'    "{val}": True,')
    print("}")

    print(f"\nTotal unique motifs: {len(unique)}")

if __name__ == "__main__":
    main()