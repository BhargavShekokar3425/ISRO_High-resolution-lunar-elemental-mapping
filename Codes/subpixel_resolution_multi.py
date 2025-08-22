import cupy as cp
import numpy as np
import pandas as pd
from tqdm import tqdm

def create_lunar_grid_cupy(resolution):
    latitudes = np.arange(-90, 90, resolution)
    longitudes = np.arange(-180, 180, resolution)

    num_rows = len(latitudes)
    num_cols = len(longitudes)

    # Initialize grid arrays on GPU
    grid_shape = (num_rows, num_cols)
    counts = cp.zeros(grid_shape, dtype=cp.int32)

    elements = ['al/si', 'mg/si', 'ca/si', 'mn/si',
                'cr/si', 'ti/si', 'fe_k/si', 'fe_l/si']
    element_arrays = {element: cp.zeros(grid_shape, dtype=cp.float32) for element in elements}

    return latitudes, longitudes, counts, element_arrays, num_rows, num_cols

def update_grid_with_abundance_cupy(latitudes, longitudes, counts, element_arrays, update_df, resolution, num_rows, num_cols):
    lat_min = -90
    lon_min = -180
    elements = list(element_arrays.keys())

    # Convert update_df to CuPy arrays
    update_data_array = cp.asarray(update_df[['u_l_lon', 'u_r_lon', 'b_r_lon', 'b_l_lon',
                                              'u_l_lat', 'u_r_lat', 'b_r_lat', 'b_l_lat'] + elements].values)

    # Precompute grid indices for all data points
    lons = update_data_array[:, 0:4]
    lats = update_data_array[:, 4:8]
    element_values = update_data_array[:, 8:]

    min_lats = cp.min(lats, axis=1)
    max_lats = cp.max(lats, axis=1)
    min_lons = cp.min(lons, axis=1)
    max_lons = cp.min(lons, axis=1)

    i_mins = cp.floor((min_lats - lat_min) / resolution).astype(cp.int32)
    i_maxs = cp.floor((max_lats - lat_min) / resolution).astype(cp.int32)
    j_mins = cp.floor((min_lons - lon_min) / resolution).astype(cp.int32)
    j_maxs = cp.floor((max_lons - lon_min) / resolution).astype(cp.int32)

    # Ensure indices are within grid bounds
    i_mins = cp.clip(i_mins, 0, num_rows - 1)
    i_maxs = cp.clip(i_maxs, 0, num_rows - 1)
    j_mins = cp.clip(j_mins, 0, num_cols - 1)
    j_maxs = cp.clip(j_maxs, 0, num_cols - 1)

    # Process each data point
    for idx in tqdm(range(update_data_array.shape[0]), desc='Updating grid'):
        i_min = int(i_mins[idx].item())
        i_max = int(i_maxs[idx].item())
        j_min = int(j_mins[idx].item())
        j_max = int(j_maxs[idx].item())
        region_element_values = element_values[idx]

        # Prepare slices
        i_indices = cp.arange(i_min, i_max + 1)
        j_indices = cp.arange(j_min, j_max + 1)
        ii, jj = cp.meshgrid(i_indices, j_indices, indexing='ij')

        # Update counts
        counts[ii, jj] += 1

        # Update elemental abundances
        counts_grid = counts[ii, jj]
        previous_counts = counts_grid - 1
        previous_counts = cp.where(previous_counts == 0, 1, previous_counts)

        for k, element in enumerate(elements):
            grid_values = element_arrays[element][ii, jj]
            region_value = region_element_values[k]

            # Update the grid's elemental abundance using the averaging formula
            new_values = (previous_counts * grid_values + region_value) / counts_grid
            element_arrays[element][ii, jj] = new_values


def save_grid_to_csv_cupy(latitudes, longitudes, counts, element_arrays, resolution):
    num_rows, num_cols = counts.shape
    total_cells = num_rows * num_cols

    # Move data from GPU to CPU
    counts_cpu = cp.asnumpy(counts)
    element_arrays_cpu = {element: cp.asnumpy(array) for element, array in element_arrays.items()}

    # Prepare data for DataFrame
    grid_data = {
        'u_l_lat': np.repeat(latitudes, num_cols),
        'u_l_lon': np.tile(longitudes, num_rows),
        'counts': counts_cpu.flatten()
    }

    for element in element_arrays_cpu:
        grid_data[element] = element_arrays_cpu[element].flatten()

    grid_df = pd.DataFrame(grid_data)

    # Compute the other corner coordinates
    grid_df['u_r_lat'] = grid_df['u_l_lat']
    grid_df['u_r_lon'] = grid_df['u_l_lon'] + resolution
    grid_df['b_r_lat'] = grid_df['u_l_lat'] + resolution
    grid_df['b_r_lon'] = grid_df['u_l_lon'] + resolution
    grid_df['b_l_lat'] = grid_df['u_l_lat'] + resolution
    grid_df['b_l_lon'] = grid_df['u_l_lon']

    grid_df.to_csv(f"subpixel_resolution_at_{resolution}.csv", index=False)

# Example usage
if __name__ == "__main__":
    resolution = 0.1
    print("Creating grid...")
    latitudes, longitudes, counts, element_arrays, num_rows, num_cols = create_lunar_grid_cupy(resolution)
    print("Grid created.")

    print("Loading update data...")
    update_data = pd.read_csv("catalogue_with_updated_solar_flare_data_clipped.csv")
    print("Update data loaded.")

    print("Updating grid with abundances...")
    update_grid_with_abundance_cupy(latitudes, longitudes, counts, element_arrays, update_data, resolution, num_rows, num_cols)
    print("Grid update complete.")

    print("Saving grid to CSV...")
    save_grid_to_csv_cupy(latitudes, longitudes, counts, element_arrays, resolution)
    print("Grid saved.")
