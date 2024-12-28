import os
import pandas as pd
import sys

def save_to_hdf(base_dir, hdf_file):
    with pd.HDFStore(hdf_file, 'w') as store:
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file.endswith('.csv'):
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, base_dir)
                    df = pd.read_csv(file_path)
                    store.put(rel_path.replace(os.sep, '/').replace('.csv', ''), df)
                    print(f"Saved {rel_path} to HDF5 file")

def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <base_directory> <output_hdf5_file>")
        sys.exit(1)

    base_dir = sys.argv[1]
    hdf_file = sys.argv[2]

    save_to_hdf(base_dir, hdf_file)
    print(f"All data from {base_dir} has been saved to {hdf_file}")

if __name__ == "__main__":
    main()

