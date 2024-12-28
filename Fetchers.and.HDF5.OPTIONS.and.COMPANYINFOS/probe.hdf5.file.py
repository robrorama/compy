import pandas as pd

def view_hdf5_contents(hdf_file):
    with pd.HDFStore(hdf_file, 'r') as store:
        print("Contents of the HDF5 file:")
        for key in store.keys():
            print(key)
            df = store.get(key)
            print(df.head())  # Display the first few rows of each DataFrame

def main():
    hdf_file = "DERBY.FIRST.h5"  # Replace with the path to your HDF5 file
    view_hdf5_contents(hdf_file)

if __name__ == "__main__":
    main()

