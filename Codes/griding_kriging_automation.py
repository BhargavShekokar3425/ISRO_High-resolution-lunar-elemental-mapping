import mysql.connector
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from pykrige.ok import OrdinaryKriging
subpixel_to_do = True

def update_ratios_in_mysql(input_list, passwrd, datab, prt):
    """
    Updates MySQL database ratios using centroid location and weighted mean calculation.
    Then, performs kriging on the 10 nearest neighbors with zero centroid counts and updates another table.

    :param input_list: List containing centroid lat, lon, and 8 ratio values
    """
    table_name = 'grid_data'  # Replace with your table name

    # Extract centroid and ratios from input list
    centroid_lat, centroid_lon = input_list[0], input_list[1]
    input_ratios = input_list[2:10]

    # Connect to the MySQL database
    conn2 = mysql.connector.connect(
        host='localhost',
        user='root',
        password=passwrd,
        database=datab,
        port = prt
    )
    cursor2 = conn2.cursor()

    # SQL query to check where the centroid lies in the defined area
    select_query = f"""
    SELECT 
        u_l_lat, u_l_lon, u_r_lat, u_r_lon, b_r_lat, b_r_lon, b_l_lat, b_l_lon, 
        `al/si`, `mg/si`, `ca/si`, `mn/si`, `cr/si`, `ti/si`, `fe_k/si`, `fe_l/si`, centroid_counts
    FROM {table_name}
    WHERE %s BETWEEN b_r_lat AND u_l_lat
    AND %s BETWEEN u_l_lon AND b_r_lon;
    """

    cursor2.execute(select_query, (centroid_lat, centroid_lon))
    result = cursor2.fetchone()

    if result:
        # Extract SQL data
        sql_coords = result[0:8]
        sql_ratios = result[8:16]
        centroid_count = result[16]

        # Calculate updated ratios using weighted mean
        updated_ratios = [
            (1 * input_ratios[i] + centroid_count * sql_ratios[i]) / (1 + centroid_count)
            for i in range(8)
        ]

        # Update the SQL table with new ratios and increment centroid_counts
        update_query = f"""
        UPDATE {table_name}
        SET 
            `al/si` = %s, `mg/si` = %s, `ca/si` = %s, `mn/si` = %s, 
            `cr/si` = %s, `ti/si` = %s, `fe_k/si` = %s, `fe_l/si` = %s, 
            centroid_counts = centroid_counts + 1
        WHERE ROUND(u_l_lat, 1) = %s AND ROUND(u_l_lon, 1) = %s AND
    ROUND(u_r_lat, 1) = %s AND ROUND(u_r_lon, 1) = %s AND
    ROUND(b_r_lat, 1) = %s AND ROUND(b_r_lon, 1) = %s AND
    ROUND(b_l_lat, 1) = %s AND ROUND(b_l_lon, 1) = %s;
        """
        cursor2.execute(update_query, (*updated_ratios, *sql_coords))
        conn2.commit()

        print("grid data updated for", sql_coords)

    # Read 'grid_data' and 'kriged_data' into pandas DataFrames
    df = pd.read_sql(f"SELECT * FROM {table_name}", conn2)
    df1 = pd.read_sql("SELECT * FROM kriged_data", conn2)

    # Compute centroid positions for each grid cell
    df['centroid_lat'] = df[['u_l_lat', 'u_r_lat', 'b_r_lat', 'b_l_lat']].mean(axis=1)
    df['centroid_lon'] = df[['u_l_lon', 'u_r_lon', 'b_r_lon', 'b_l_lon']].mean(axis=1)
    df1['centroid_lat'] = df['centroid_lat']
    df1['centroid_lon'] = df['centroid_lon']

    # Extract the points with centroid_counts == 0
    zero_centroid_df = df[df['centroid_counts'] == 0].copy()

    # Coordinates of the updated point
    updated_point = np.array([[centroid_lat, centroid_lon]])

    # Extract coordinates from zero_centroid_df
    coords = zero_centroid_df[['centroid_lat', 'centroid_lon']].values

    # Use NearestNeighbors to find the 10 nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(coords)
    distances, indices = nbrs.kneighbors(updated_point)

    # Get the indices of the 10 nearest neighbors
    nearest_indices = zero_centroid_df.iloc[indices[0]].index

    # Get the 10 nearest neighbors data
    nearest_neighbours = zero_centroid_df.loc[nearest_indices].copy()

    # Ratio columns
    ratio_columns = ['al/si', 'mg/si', 'ca/si', 'mn/si', 'cr/si', 'ti/si', 'fe_k/si', 'fe_l/si']

    # Get the known data points (centroid_counts > 0)
    known_data_df = df[df['centroid_counts'] > 0].copy()

    # Perform kriging for each ratio
    for ratio_col in ratio_columns:
        # Define a latitude and longitude range around the unknown point
        lat_range = (centroid_lat - 10, centroid_lat + 10)
        lon_range = (centroid_lon - 10, centroid_lon + 10)

        # Filter the known data points to only include those within the defined range
        subset_known_data_df = known_data_df[
            (known_data_df['centroid_lat'] >= lat_range[0]) & (known_data_df['centroid_lat'] <= lat_range[1]) &
            (known_data_df['centroid_lon'] >= lon_range[0]) & (known_data_df['centroid_lon'] <= lon_range[1])
        ]

        # Check if the subset is empty (fallback to random sampling if necessary)
        if len(subset_known_data_df) == 0:
            print(f"No known points found in the range {lat_range}, {lon_range}. Using a random sample.")
            subset_known_data_df = known_data_df.sample(n=min(10000, len(known_data_df)), random_state=42)

        # Coordinates and ratio values of the filtered known data points
        known_coords = subset_known_data_df[['centroid_lat', 'centroid_lon']].values
        known_values = subset_known_data_df[ratio_col].values

        # Remove NaN values
        mask = ~np.isnan(known_values)
        known_coords = known_coords[mask]
        known_values = known_values[mask]

        # Coordinates where we want to estimate the ratio
        unknown_coords = nearest_neighbours[['centroid_lat', 'centroid_lon']].values

        # Perform ordinary kriging
        try:
            OK = OrdinaryKriging(
                known_coords[:, 1], known_coords[:, 0], known_values,
                variogram_model='exponential', verbose=False, enable_plotting=False
            )
            z, ss = OK.execute('points', unknown_coords[:, 1], unknown_coords[:, 0])

            # Update the 'kriged_data' DataFrame 'df1' with these values
            df1.loc[nearest_indices, ratio_col] = z

        except Exception as e:
            print(f"Error in kriging for ratio {ratio_col}: {e}")


    # Update the 'kriged_data' table in SQL with the updated 'df1'
    ratio_columns = ['al/si_kriged', 'mg/si_kriged', 'ca/si_kriged', 'mn/si_kriged', 'cr/si_kriged', 'ti/si_kriged', 'fe_k/si_kriged']
    for idx in nearest_indices:
        # Get the updated ratio values
        updated_ratios = df1.loc[idx, ratio_columns].values.tolist()
        # Get the corresponding centroid_lat and centroid_lon
        centroid_lat = df1.loc[idx, 'centroid_lat']
        centroid_lon = df1.loc[idx, 'centroid_lon']
        ullat = df1.loc[idx, 'u_l_lat']
        ullon = df1.loc[idx, 'u_l_lon']
        urlat = df1.loc[idx, 'u_r_lat']
        urlon = df1.loc[idx, 'u_r_lon']
        brlat = df1.loc[idx, 'b_r_lat']
        brlon = df1.loc[idx, 'b_r_lon']
        bllat = df1.loc[idx, 'b_l_lat']
        bllon = df1.loc[idx, 'b_l_lon']
        # Prepare the SQL update query using centroid_lat and centroid_lon
        update_query = f"""
        UPDATE kriged_data
        SET 
            `al/si_kriged` = %s, `mg/si_kriged` = %s, `ca/si_kriged` = %s, `mn/si_kriged` = %s, 
            `cr/si_kriged` = %s, `ti/si_kriged` = %s, `fe_k/si_kriged` = %s
        WHERE ROUND(cent_lat, 1) = %s AND ROUND(cent_lon, 1) = %s;
        """

        cursor2.execute(update_query, (*updated_ratios, centroid_lat, centroid_lon))
        conn2.commit()
        insert_query = f"""
        INSERT INTO update_data 
        (`u_l_lat`,`u_l_lon`,`u_r_lat`,`u_r_lon`,`b_r_lat`,`b_r_lon`,`b_l_lat`,`b_l_lon`,`cent_lat`, `cent_lon`, `al/si_kriged`, `mg/si_kriged`, `ca/si_kriged`, `mn/si_kriged`, 
        `cr/si_kriged`, `ti/si_kriged`, `fe_k/si_kriged`)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        """
        # Execute the update query with centroid_lat and centroid_lon
        cursor2.execute(insert_query, (ullat, ullon, urlat, urlon, brlat, brlon, bllat, bllon, centroid_lat, centroid_lon, *updated_ratios))
        conn2.commit()
    print("kriging data added for", unknown_coords)
    if subpixel_to_do:
        print("Performing subpixel resolution data update")

        coord = input_list[10:18]
        max_lat = max(coord[0:4])
        min_lat = min(coord[0:4])
        max_long = max(coord[4:8])
        min_long = min(coord[4:8])

        ratio_columns = ['al/si', 'mg/si', 'ca/si', 'mn/si', 'cr/si', 'ti/si', 'fe_k/si', 'fe_l/si']

        select_query = """
        SELECT *
        FROM subpixel_resolution_data
        WHERE (`u_l_lat` > %s) AND (`b_l_lat` < %s)
        AND (`u_l_lon` > %s) AND (u_r_lon < %s);
        """
        cursor2.execute(select_query, (min_lat, max_lat, min_long, max_long))
        selected_rows = cursor2.fetchall()

        print("min max lat long:-", min_lat, max_lat, min_long, max_long)
        update_query = """
        UPDATE subpixel_resolution_data
        SET `al/si` = %s, `mg/si` = %s, `ca/si` = %s, `mn/si` = %s, `cr/si` = %s, `ti/si` = %s, `fe_k/si` = %s, `fe_l/si` = %s, `counts` = %s
        WHERE ROUND(u_l_lat,1) = %s AND ROUND(u_l_lon,1) = %s AND ROUND(u_r_lat,1) = %s AND ROUND(u_r_lon,1) = %s AND ROUND(b_r_lat,1) = %s AND ROUND(b_r_lon,1) = %s AND ROUND(b_l_lat,1) = %s AND ROUND(b_l_lon,1) = %s;
        """
        columns = [desc[0] for desc in cursor2.description]
        df = pd.DataFrame(selected_rows, columns=columns)

        for i in range(len(df)):
            j = 0
            cnts = df.loc[i, "counts"]
            for col in ratio_columns:
                df[col] = (df[col]*cnts + input_ratios[j])/(cnts+1)
                j+=1
            print("updating subpixel for coords:-", (df.at[i, "u_l_lat"], df.at[i, "u_l_lon"], df.at[i, "u_r_lat"], df.at[i, "u_r_lon"], df.at[i, "b_r_lat"], df.at[i, "b_r_lon"], df.at[i, "b_l_lat"], df.at[i, "b_l_lon"]), "from", df.at[i, "counts"], "to", df.at[i, "counts"]+1)
            
            cursor2.execute(update_query, (df.at[i, "al/si"], df.at[i, "mg/si"], df.at[i, "ca/si"], df.at[i, "mn/si"], df.at[i, "cr/si"], df.at[i, "ti/si"], df.at[i, "fe_k/si"], df.at[i, "fe_l/si"], df.at[i, "counts"]+1, df.at[i, "u_l_lat"], df.at[i, "u_l_lon"], df.at[i, "u_r_lat"], df.at[i, "u_r_lon"], df.at[i, "b_r_lat"], df.at[i, "b_r_lon"], df.at[i, "b_l_lat"], df.at[i, "b_l_lon"]))
            
            conn2.commit()

    cursor2.close()
    conn2.close()