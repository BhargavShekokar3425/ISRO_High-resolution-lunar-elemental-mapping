import os
import numpy as np
from astropy.io import fits
import pandas as pd
from datetime import datetime, timedelta
import mysql.connector

# Function to read light curve data from a .lc file
def read_lc(file_path):
    try:
        with fits.open(file_path) as lc:
            time = lc[1].data['TIME']  # Time in MET
            rate = lc[1].data['RATE']  # Count rate in counts/sec
        return time, rate
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None, None

# Function to group high flare intervals
def get_high_flare_intervals(time, rate, threshold, min_duration=8):
    high_indices = np.where(rate > threshold)[0]
    if len(high_indices) == 0:
        return []

    intervals = []
    start = time[high_indices[0]]
    for i in range(1, len(high_indices)):
        if high_indices[i] != high_indices[i - 1] + 1:  # Gap in indices
            end = time[high_indices[i - 1]]
            if end - start > min_duration:
                intervals.append((start, end))
            start = time[high_indices[i]]
    end = time[high_indices[-1]]
    if end - start > min_duration:
        intervals.append((start, end))
    return intervals

# Function to convert MET to UTC
def met_to_utc(met_time):
    ref_time = datetime(2017, 1, 1)
    return ref_time + timedelta(seconds=met_time)

# Process all .lc files in the directory structure
def process_monthly_data(base_directory, passwrd, datab, prt, threshold_multiplier=2, min_duration=8):
    results = []

    # Traverse years
    for year in sorted(os.listdir(base_directory)):
        year_path = os.path.join(base_directory, year)
        if not os.path.isdir(year_path):
            continue

        # Traverse months
        for month in sorted(os.listdir(year_path)):
            month_path = os.path.join(year_path, month)
            if not os.path.isdir(month_path):
                continue

            all_rates = []
            file_data = []

            # Traverse days within the month
            for day in sorted(os.listdir(month_path)):
                day_path = os.path.join(month_path, day)
                if not os.path.isdir(day_path):
                    continue

                # Use os.walk to recursively find .lc files in all subdirectories
                for root, _, files in os.walk(day_path):
                    for file_name in files:
                        if file_name.endswith(".lc"):
                            file_path = os.path.join(root, file_name)
                            time, rate = read_lc(file_path)
                            if time is not None and rate is not None:
                                all_rates.extend(rate)
                                file_data.append((file_name, time, rate))

            # Calculate statistics for the month
            if all_rates:
                all_rates = np.array(all_rates)
                mean_rate = np.mean(all_rates)
                std_rate = np.std(all_rates)
                threshold = mean_rate + threshold_multiplier * std_rate
                print(f"Processed {year}/{month}: Mean={mean_rate}, Std={std_rate}, Threshold={threshold}")

                # Identify high flare intervals
                for file_name, time, rate in file_data:
                    intervals = get_high_flare_intervals(time, rate, threshold, min_duration)
                    for start, end in intervals:
                        duration = end - start
                        results.append({
                            "File Name": file_name,
                            "Date": f"{year}-{month}",
                            "Start_Time": met_to_utc(start).strftime("%Y-%m-%d %H:%M:%S"),
                            "End_Time": met_to_utc(end).strftime("%Y-%m-%d %H:%M:%S"),
                            "Duration": duration
                        })
            else:
                print(f"No valid data for {year}/{month}")

    # Establish the MySQL connection (change the placeholders with your details)
    connection2 = mysql.connector.connect(
        host='localhost',
        user='root',
        password=passwrd,
        database=datab,
        port=prt
    )

    # Create a cursor object
    cursor2 = connection2.cursor()

    # SQL insert query
    insert_query = "INSERT INTO high_solar_flare_intervals (file_name, date, start_time, end_time, duration) VALUES (%s, %s, %s, %s, %s)"

    # Save results to a CSV
    if results:
        print(results)
        for entry in results:
            # Extract values from the dictionary
            file_name = entry["File Name"]
            date = entry["Date"]
            start_time = entry["Start_Time"]
            end_time = entry["End_Time"]
            duration = entry["Duration"]
            cursor2.execute(insert_query, (file_name, date, start_time, end_time, duration))
            print("entry:-", entry)
            # Commit the transaction to the database
        connection2.commit()
    cursor2.close()
    connection2.close()
