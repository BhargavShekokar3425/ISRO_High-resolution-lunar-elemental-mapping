import mysql.connector
import csv
import pandas as pd

# Database configuration for the MySQL server
server_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '1234',
    'port':3306

}

# Function to create a database
def create_database(db_name):
    connection = None  # Initialize connection variable
    try:
        connection = mysql.connector.connect(**server_config)
        cursor = connection.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_name}")
        print(f"Database '{db_name}' created successfully.")
    except mysql.connector.Error as error:
        print(f"Error creating database: {error}")
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()


def infer_mysql_datatype(value):
    """Infer MySQL data type based on Python type."""
    """try:
        int(value)
        return "INT"
    except ValueError:
        pass"""

    try:
        float(value)
        return "FLOAT"
    except ValueError:
        pass

    # Default to VARCHAR for non-numeric values
    return "VARCHAR(255)"

def specific(value):
    return "VARCHAR(255)"

def create_table_structure_in_db(csv_file_path, db_name, table_name, columns_to_keep=None):
    connection = None  # Initialize connection variable
    try:
        db_config = server_config.copy()
        db_config['database'] = db_name
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()

        # Read the CSV file with pandas
        data = pd.read_csv(csv_file_path)

        # Filter columns to keep
        if columns_to_keep:
            data = data[columns_to_keep]

        # Standardize column names
        standardized_columns = {col: col.replace(" ", "_").replace("-", "_") for col in data.columns}
        data.rename(columns=standardized_columns, inplace=True)

        # Infer data types
        mysql_data_types = {}
        for column in data.columns:
            first_non_null_value = data[column].dropna().iloc[0] if not data[column].dropna().empty else ""
            if table_name == "geotail": mysql_data_types[column] = specific(first_non_null_value)
            else: mysql_data_types[column] = infer_mysql_datatype(first_non_null_value)

        # Construct CREATE TABLE query
        columns = ", ".join([f"`{col}` {dtype}" for col, dtype in mysql_data_types.items()])
        create_query = f"CREATE TABLE IF NOT EXISTS `{table_name}` ({columns});"
        cursor.execute(create_query)

        connection.commit()
        print(f"Table '{table_name}' created in database '{db_name}' with dynamic data types.")
    except Exception as error:
        print(f"Error creating table: {error}")
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()



def populate_table_from_csv(csv_file_path, db_name, table_name, columns_to_keep=None):
    try:
        db_config = server_config.copy()
        db_config['database'] = db_name
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()

        with open(csv_file_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            headers = next(csv_reader)

            # Standardize column names
            standardized_headers = [header.replace(" ", "_").replace("-", "_") for header in headers]

            # Filter columns to keep
            if columns_to_keep:
                columns_to_keep = [col.strip() for col in columns_to_keep]
                headers = [header for header in standardized_headers if header in columns_to_keep]

            placeholders = ', '.join(['%s'] * len(headers))
            insert_query = f"INSERT INTO `{table_name}` ({', '.join([f'`{header}`' for header in headers])}) VALUES ({placeholders})"

            for row in csv_reader:
                filtered_row = [value for header, value in zip(headers, row) if header in columns_to_keep]
                cursor.execute(insert_query, filtered_row)

        connection.commit()
        print(f"Data populated successfully into table '{table_name}'.")
    except mysql.connector.Error as error:
        print(f"Error populating table: {error}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()


# Main function
def main():
    # Database name
    db_name = 'isro_team_73'

    # Create the database
    create_database(db_name)
        #{'file': 'catalogue_with_updated_solar_flare_data.csv', 'table': 'catalogue', 'columns_to_keep': ['file_name','u_l_lat','u_r_lat','b_r_lat','b_l_lat','u_l_lon','u_r_lon','b_r_lon','b_l_lon','start_date','end_date','start_time','end_time','angle','side','flare','geotail','al_max_counts','si_max_counts','mg_max_counts','ca_max_counts','mn_max_counts','cr_max_counts','ti_max_counts','fe_k_max_counts','fe_l_max_counts','al_energy','si_energy','mg_energy','ca_energy','mn_energy','cr_energy','ti_energy','fe_k_energy','fe_l_energy','al_area','si_area','mg_area','ca_area','mn_area','cr_area','ti_area','fe_k_area','fe_l_area','al_abundance_relative','si_abundance_relative','mg_abundance_relative','ca_abundance_relative','mn_abundance_relative','cr_abundance_relative','ti_abundance_relative','fe_k_abundance_relative','fe_l_abundance_relative','al/si','mg/si','ca/si','mn/si','cr/si','ti/si','fe_k/si','fe_l/si'], 'populate': False},
        #{'file': 'grid_results_new_with_solar_flare_clipped.csv', 'table': 'grid_data', 'columns_to_keep': ['u_l_lat','u_l_lon','u_r_lat','u_r_lon','b_r_lat','b_r_lon','b_l_lat','b_l_lon','al/si','mg/si','ca/si','mn/si','cr/si','ti/si','fe_k/si','fe_l/si','centroid_counts'], 'populate': True},
        #{'file': 'kriged_results_kriging_ratios_1.csv', 'table': 'kriged_data', 'columns_to_keep': ['u_l_lat','u_l_lon','u_r_lat','u_r_lon','b_r_lat','b_r_lon','b_l_lat','b_l_lon','cent_lat','cent_lon','al/si_kriged','mg/si_kriged','ca/si_kriged','mn/si_kriged','cr/si_kriged','ti/si_kriged','fe_k/si_kriged','fe_l/si_kriged'], 'populate': True},
        #{'file': 'background_mean_sigma_counts.csv', 'table': 'background_mean_sigma_counts', 'columns_to_keep': ['date','al_mean','mg_mean','si_mean','ca_mean','mn_mean','cr_mean','ti_mean','fe_k_mean','fe_l_mean','al_sigma','mg_sigma','si_sigma','ca_sigma','mn_sigma','cr_sigma','ti_sigma','fe_k_sigma','fe_l_sigma','nightside_file_count'], 'populate': False},
        #{'file': 'high_solar_flare_intervals.csv', 'table': 'high_solar_flare_intervals', 'columns_to_keep': ['File_Name','Date','Start_Time','End_Time','Duration'], 'populate': False},
        # Create table and populate with data
        #{'file': 'geotail.csv', 'table': 'geotail', 'columns_to_keep': ['Full_Moon_Date','Start_Timestamp','End_Timestamp'], 'populate': True},"""
    # CSV file paths and options
    csv_files = [
        # Create table structure only
        
        {'file': 'subpixel_resolution_at_0.1.csv', 'table': 'subpixel_resolution_data', 'columns_to_keep': ['u_l_lat','u_l_lon','u_r_lat','u_r_lon','b_r_lat','b_r_lon','b_l_lat','b_l_lon','al/si','mg/si','ca/si','mn/si','cr/si','ti/si','fe_k/si','fe_l/si','counts'], 'populate': True}
    ]

    # Process each CSV file
    for csv_info in csv_files:
        # Step 1: Create table structure
        create_table_structure_in_db(csv_info['file'], db_name, csv_info['table'], csv_info['columns_to_keep'])

        # Step 2: Populate table if required
        if csv_info['populate']:
            populate_table_from_csv(csv_info['file'], db_name, csv_info['table'], csv_info['columns_to_keep'])

if __name__ == "__main__":
    main()
