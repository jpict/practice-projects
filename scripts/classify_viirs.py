#!/usr/bin/env python3
import argparse
import logging
import xarray as xr
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

def setup_logging(log_level):
    logging.basicConfig(
        level={
            0: logging.ERROR,
            1: logging.WARNING,
            2: logging.INFO,
            3: logging.DEBUG
        }[log_level],
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def load_data(file_path):
    try:
        data = xr.open_dataset(file_path, engine='rasterio')
        logging.info(f"Data loaded successfully from {file_path}")
        return data
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}")
        raise

def classify_data(flat_data, n_clusters=5):
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=9)
        labels = kmeans.fit_predict(flat_data)
        logging.info("Data classified successfully.")
        return labels
    except Exception as e:
        logging.error(f"Error during classification: {e}")
        raise


def main():
    # Parse into command line arguments
    parser = argparse.ArgumentParser(description="Classify VIIRS data using KMeans clustering.")
    parser.add_argument('viirs_file', type=str, help='Path to the VIIRS data file')
    parser.add_argument('--n_clusters', type=int, default=5, help='Number of clusters for KMeans')
    parser.add_argument("-v", action="count", default=0, help="Increase verbosity level (use -v, -vv, -vvv)")
    args = parser.parse_args()

    # Setup logging based on verbosity level
    setup_logging(args.v) 
    logging.info("Script started.")

    # Load data
    xds = load_data(args.viirs_file)
    
    # Extract data as numpy array and reshape so that it in N x bands
    logging.info("Extract and reshape data along singleton dimension.")
    n_bands = xds.band_data.shape[0]
    n_x = xds.band_data.shape[1]
    n_y = xds.band_data.shape[2]
    data = xds.band_data.transpose('y', 'x', 'band').values.reshape(-1, n_bands)

    # Classify data
    logging.info("Classifying data using KMeans.")
    results = classify_data(data, args.n_clusters)

    # Reshape results back to original dimensions
    logging.info("Reshaping results to match original dimensions.")
    results = results.reshape(n_x, n_y).T

    # Add results to dataset
    xds['clusters'] = xr.DataArray(results, dims=('x', 'y'))

    # Plot both results
    logging.info("Plotting results.")
    plt.figure(figsize=(10, 10))
    xds['clusters'].plot.imshow(cmap='tab10', add_colorbar=False)
    plt.show()

    logging.info("Script finished.")

if __name__ == "__main__":
    main()