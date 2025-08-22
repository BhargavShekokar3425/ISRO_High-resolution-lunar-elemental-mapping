from astropy.io import fits
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import warnings
from tqdm import tqdm
import os
import mysql.connector
import background_generation
import background_mean_sigma_counts
import catalogue_generator
import xsm_data_generator   
import shutil
import time
import logging
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
    handlers=[
        logging.FileHandler("processing.log"),  # Log to a file
        logging.StreamHandler()  # Log to the console
    ]
)


def remove_subdirectories(directory_path):
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)

def fits_extractor(fits_dir):
    file_list = os.listdir(fits_dir)
    path = fits_dir
    for file in file_list:
        file_path = path+'/'+file
        if file.endswith(".fits"): 
            fits_files.append(file_path)
        elif file.endswith(".xml"): 
            os.remove(file_path)
        else:
            fits_extractor(file_path)
    return

def xsm_extractor(xsm_dir):
    file_list = os.listdir(xsm_dir)
    path = xsm_dir
    for file in file_list:
        file_path = path+'/'+file
        if file.endswith(".lc"):
            timestamp = file[0:-3]
            date = str(timestamp).split('_')[2]
            year = date[0:4]
            month = date[4:6]
            day = date[6:8]
            dir = path+'/'+year+'/'+month+'/'+day
            os.makedirs(dir, exist_ok=True)
            shutil.move(file_path, dir)

def xsm_extractor2(xsm_dir):
    file_list = os.listdir(xsm_dir)
    path = xsm_dir
    for file in file_list:
        file_path = path+'/'+file
        if file.endswith(".lc"): xsm_files.append(file_path)
        else: xsm_extractor2(file_path)
    return

passwd = "1234"
db = "isro_team_73"
port_con = 3306

# Step 1: Establish a connection to your database
connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password=passwd,
    database=db,
    port = port_con
)

cursor = connection.cursor()

# Step 2: Write the SQL query to fetch data
query1 = "SELECT * FROM background_mean_sigma_counts"

# Step 3: Use pandas to read the SQL table into a DataFrame
background_data = pd.read_sql(query1, connection)
bgmsc_dates = []
for i in range(len(background_data)):
    bgmsc_dates.append(background_data.loc[i, "date"])


new_data_dir = "New_Data"
fits_file_dir = "fits_files"
xsm_file_dir = "xsm_files"
move_dir = "processed data"
night_dir = "fits_files/nightside"
day_dir = "fits_files/dayside"
xsm_move_dir = "xsm_files"

fits_files = []
xsm_files = []

column_names = ["date", "al_mean", "mg_mean", "si_mean", "ca_mean", "mn_mean", "cr_mean", 
                        "ti_mean", "fe_k_mean", "fe_l_mean", "al_sigma", "mg_sigma", "si_sigma", 
                        "ca_sigma", "mn_sigma", "cr_sigma", "ti_sigma", "fe_k_sigma", "fe_l_sigma", "nightside_file_count"]  # Column names

while(True):

    print("Extracting fits files")
    fits_extractor(f"{new_data_dir}/{fits_file_dir}")
    print("fits files extracted")
    print("Extrcting xsm files")
    xsm_extractor(f"{new_data_dir}/{xsm_file_dir}")
    xsm_files = []
    xsm_extractor2(f"{new_data_dir}/{xsm_file_dir}")
    print("xsm files extracted")

    print("fits_files gathered:-",len(fits_files))
    print("xsm_files gathered:-", len(xsm_files))

    print("processing xsm data")
    xsm_data_generator.process_monthly_data(f"{new_data_dir}/{xsm_file_dir}", passwd, db, port_con)
    for file_path_xsm in xsm_files:
        file_name = str(file_path_xsm).split('/')[-1]
        date = str(file_name).split('_')[2]
        year = date[0:4]
        month = date[4:6]
        day = date[6:8]
        to_move_dir = move_dir+'/'+xsm_move_dir+'/'+year+'/'+month+'/'+day
        ddir = move_dir+'/'+day_dir+'/'+year+'/'+month+'/'+day
        os.makedirs(to_move_dir, exist_ok=True)
        shutil.move(file_path_xsm, to_move_dir)
        if os.path.exists(ddir):
            file_list_for_xsm_update = os.listdir(ddir)
            for file in file_list_for_xsm_update:
                file_path = ddir+'/'+file
                catalogue_generator.update(file_path, passwd, db, port_con)

    remove_subdirectories(f"{new_data_dir}/{xsm_file_dir}")

    for file_path in fits_files:
        try:
            with fits.open(file_path) as hdul:
                for x, hdu in enumerate(hdul):
                    header = hdu.header
                    angle = header.get('solarang')
                    st_time = header.get('startime')
        except:
            continue

        st_date = str(st_time).split('T')[0]
        side = "dayside" if angle <= 90 else "nightside"

        year = str(st_date).split('-')[0]
        month = str(st_date).split('-')[1]
        day = str(st_date).split('-')[2]

        ndir = move_dir+'/'+night_dir+'/'+year+'/'+month+'/'+day
        ddir = move_dir+'/'+day_dir+'/'+year+'/'+month+'/'+day

        print(side)
        if side == "nightside":
            found = 0
            if st_date in bgmsc_dates: found = 1

            if found == 0:
                print("generating background")
                background_generation.result(file_path = file_path, count=0)
                values = background_mean_sigma_counts.output(st_date)
                counts = 0
                bgmsc_dates.append(st_date)
            else:
                print("generating background_mean_sigma")
                count_access_query = f"SELECT nightside_file_count FROM background_mean_sigma_counts WHERE date = '{st_date}'"
                cursor.execute(count_access_query)
                counts = cursor.fetchall()
                counts = counts[0][0]
                print("generating background")
                background_generation.result(file_path = file_path, count = counts)
                print("generating mean and sigma")
                values = background_mean_sigma_counts.output(st_date)
                delete_query = f"DELETE FROM background_mean_sigma_counts WHERE date = '{st_date}'"
                cursor.execute(delete_query)
                connection.commit()

            values.append(counts+1)

            print("updating mean sigma")
            try:
                table_name = "background_mean_sigma_counts"  # Replace with your table name

                # Construct the SQL INSERT query
                columns = ", ".join(column_names)
                placeholders = ", ".join(["%s"] * len(column_names))  # Using %s for parameterized queries
                insert_query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
                cursor.execute(insert_query, values)
                connection.commit()
            except:
                continue

            os.makedirs(ndir, exist_ok=True)
            shutil.move(file_path, ndir)

            if (found == 1) and (os.path.isdir(ddir)):
                print("updating catalogue")
                rectify_list = os.listdir(ddir)
                for rect_file in rectify_list:
                    rect_path = ddir+'/'+rect_file
                    catalogue_generator.update(rect_path, passwd, db, port_con)

        else:
            if st_date not in bgmsc_dates: continue
            print("generating catalogue")
            is_done = catalogue_generator.insert(file_path, passwd, db, port_con)
            if is_done:
                os.makedirs(ddir, exist_ok=True)
                shutil.move(file_path, ddir)

    print("removing fits directories")
    if len(fits_files) == 0: remove_subdirectories(f"{new_data_dir}/{fits_file_dir}")

    fits_files.clear()
    xsm_files.clear()
    time.sleep(2)