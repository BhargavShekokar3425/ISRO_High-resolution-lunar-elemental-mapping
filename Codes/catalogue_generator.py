from astropy.io import fits
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import warnings
from tqdm import tqdm
from numba import jit, prange
import mysql.connector
import griding_kriging_automation
warnings.filterwarnings('ignore')

# Helper function to ensure native-endian arrays for Numba compatibility
def to_native_endian(array, dtype=np.float64):
    if array.dtype.byteorder not in ('=', '|'):  # Check if the array is non-native endian
        return array.astype(dtype)
    return array

# Power law function

def power_law(energy, a, b):
    return a * energy ** (-b)

# Function for applying arf file data on raw spectrum

def calculate_corrected_counts(counts, effective_area):
    corrected_counts = np.zeros_like(counts)
    for i in range(len(counts)):
        corrected_counts[i] = counts[i] / effective_area[i] if effective_area[i] != 0 else 0
    return corrected_counts

# Function for getting the peak value of counts for a range of energy

def find_max_in_range(energy_bins, counts, energy_range):
    mask = (energy_bins >= energy_range[0]) & (energy_bins <= energy_range[1])
    if np.any(mask):
        max_index = np.argmax(counts[mask])
        max_energy = energy_bins[mask][max_index]
        max_count = counts[mask][max_index]
        return max_energy, max_count
    else:
        return None, None

# Calculate the flux for a particular range of energy

def area_calculate(energy_bins, counts, energy_range):
    mask = (energy_bins >= energy_range[0]) & (energy_bins <= energy_range[1])
    area_data = counts[mask]
    area = np.sum(area_data)
    return area

def output(file_path, passwrd, datab, prt):

    connection2 = mysql.connector.connect(
        host="localhost",
        user="root",
        password=passwrd,
        database=datab,
        port = prt
    )

    cursor2 = connection2.cursor()
    
    query_1 = "SELECT * FROM background_mean_sigma_counts"
    query_2 = "SELECT * FROM high_solar_flare_intervals"
    query_3 = "SELECT * FROM geotail"

    all_bg_data = pd.read_sql(query_1, connection2)
    data = pd.read_sql(query_2, connection2)
    geotail = pd.read_sql(query_3, connection2)

    cursor2.close()
    connection2.close()

    bg_ms_dates = []
    for i in range(len(data)):
        bg_ms_dates.append(data.loc[i, "Date"].split()[0])

    dates = []
    for i in range(len(data)):
        dates.append(data.loc[i, "Start_Time"].split()[0])

    geotail.drop('Full_Moon_Date', axis = 1, inplace=True)
    for i in range(len(geotail)):
        geotail.loc[i, "Start_Timestamp"] = str(geotail.loc[i, "Start_Timestamp"])[0:-6]
        geotail.loc[i, "End_Timestamp"] = str(geotail.loc[i, "End_Timestamp"])[0:-6]

    geotail_dates = []
    for i in range(len(geotail)):
        st = int(geotail.loc[i, "Start_Timestamp"])
        en = int(geotail.loc[i, "End_Timestamp"])
        for j in range(st, en+1, 1000000):
            geotail_dates.append(j)

    check = 1
    file = str(file_path).split('/')[-1]

    # Initialize variables for this batch
    total_counts = None
    effective_area = None
    energy_bins = None

    # Read FITS file and extract metadata
    with fits.open(file_path) as hdul:
        for x, hdu in enumerate(hdul):
            header = hdu.header
            lat_top_left = header.get('v0_LAT')
            lat_top_right = header.get('v3_LAT')
            lat_bottom_left = header.get('v1_LAT')
            lat_bottom_right = header.get('v2_LAT')
            long_top_left = header.get('v0_LON')
            long_top_right = header.get('v3_LON')
            long_bottom_left = header.get('v1_LON')
            long_bottom_right = header.get('v2_LON')
            st_time = header.get('startime')
            end_time = header.get('endtime')
            angle = header.get('solarang')

    st_date = str(st_time).split('T')[0]
    end_date = str(end_time).split('T')[0]
    st_timestamp = str(st_time).split('T')[1]
    end_timestamp = str(end_time).split('T')[1]

    # Day/Night side determination
    side = "dayside"
    
    # Load counts from spectrum data
    with fits.open(file_path) as hdul:
        spectrum_data = hdul[1].data
        counts = spectrum_data['COUNTS']  # Assuming the FITS file has 'COUNTS' column
        total_counts = counts if total_counts is None else total_counts + counts

    # Load ARF file for effective area
    arf_file = "class_arf_v1.arf"
    with fits.open(arf_file) as hdul:
        arf_data = hdul[1].data
        effective_area = arf_data['SPECRESP']

    # Load RMF file to get energy bins
    rmf_file = "class_rmf_v1.rmf"
    with fits.open(rmf_file) as hdul:
        rmf_data = hdul[2].data
        energy_lo = rmf_data['ENERG_LO']
        energy_hi = rmf_data['ENERG_HI']
        energy_bins = (energy_lo + energy_hi) / 2

    # Filter energy range between 0.5 keV and 10 keV
    energy_mask = (energy_bins >= 0.5) & (energy_bins <= 10.0)
    filtered_counts = to_native_endian(total_counts[energy_mask])
    filtered_effective_area = to_native_endian(effective_area[energy_mask])
    filtered_energy_bins = to_native_endian(energy_bins[energy_mask])

    # Apply effective area correction
    corrected_counts = calculate_corrected_counts(filtered_counts, filtered_effective_area)

    al_peak_range, si_peak_range, mg_peak_range, ca_peak_range = (1.43, 1.53), (1.68, 1.78), (1.20, 1.30), (3.64, 3.74)
    _, al_max = find_max_in_range(filtered_energy_bins, corrected_counts, al_peak_range)
    _, si_max = find_max_in_range(filtered_energy_bins, corrected_counts, si_peak_range)
    _, mg_max = find_max_in_range(filtered_energy_bins, corrected_counts, mg_peak_range)
    _, ca_max = find_max_in_range(filtered_energy_bins, corrected_counts, ca_peak_range)

    for dt in range(len(all_bg_data)):
        if all_bg_data.loc[dt, "date"] == st_date:
            if al_max >= all_bg_data.loc[dt, "al_mean"] + 3*all_bg_data.loc[dt, "al_sigma"]:break
            elif si_max >= all_bg_data.loc[dt, "si_mean"] + 3*all_bg_data.loc[dt, "si_sigma"]:break
            elif mg_max >= all_bg_data.loc[dt, "mg_mean"] + 3*all_bg_data.loc[dt, "mg_sigma"]:break
            elif ca_max >= all_bg_data.loc[dt, "ca_mean"] + 3*all_bg_data.loc[dt, "ca_sigma"]:break
            else:
                check = 0
                break
    
    if check == 0:
        print("criteria not satisfied")
        return 1
    
    print("data:-", dates)
    flare = "low solar flare"
    if st_date in dates: flare = "high solar flare"

    sp = str(st_date).split('-')
    str_date = sp[2]+sp[1]+sp[0]
    date = int(str_date)
    if date in geotail_dates:
        geotail = 'Yes'
    else:
        geotail = 'No'
        
    try:
        bg_csv = f"background_files/{st_date}.csv"
        bg_data = pd.read_csv(bg_csv)
    except:
        return 1
    to_subtract = bg_data['0'].values
    filtered_counts_sub = to_native_endian(to_subtract[energy_mask])
    corrected_counts_sub = calculate_corrected_counts(filtered_counts_sub, filtered_effective_area)

    corrected_counts = corrected_counts-corrected_counts_sub
    corrected_counts[corrected_counts<0] = 0

    # Define excluded regions for continuum fitting
    exclude_regions = [(1.43, 1.53), (1.68, 1.78), (1.20, 1.30), (3.64, 3.74), (5.85, 5.95), (5.36, 5.46), (4.46, 4.56), (6.35, 6.45), (0.68, 0.75)]
    continuum_mask = np.ones_like(filtered_energy_bins, dtype=bool)
    for region in exclude_regions:
        region_mask = (filtered_energy_bins >= region[0]) & (filtered_energy_bins <= region[1])
        continuum_mask &= ~region_mask

    # Fit continuum with power law
    energy_for_fit = filtered_energy_bins[continuum_mask]
    counts_for_fit = corrected_counts[continuum_mask]
    try:
        popt, _ = curve_fit(power_law, energy_for_fit, counts_for_fit, p0=[1e3, 2], maxfev=2000)
        continuum_fitted = power_law(filtered_energy_bins, *popt)
    except RuntimeError as e:
        continuum_fitted = np.zeros_like(filtered_energy_bins)

    # Subtract fitted continuum
    final_counts = corrected_counts - continuum_fitted
    final_counts[final_counts < 0] = 0
    corrected_counts = final_counts

    # Define peak ranges and calculate areas and max values
    al_peak_range, si_peak_range, mg_peak_range, ca_peak_range = (1.43, 1.53), (1.68, 1.78), (1.20, 1.30), (3.64, 3.74)
    al_max_energy, al_max_count = find_max_in_range(filtered_energy_bins, corrected_counts, al_peak_range)
    si_max_energy, si_max_count = find_max_in_range(filtered_energy_bins, corrected_counts, si_peak_range)
    mg_max_energy, mg_max_count = find_max_in_range(filtered_energy_bins, corrected_counts, mg_peak_range)
    ca_max_energy, ca_max_count = find_max_in_range(filtered_energy_bins, corrected_counts, ca_peak_range)

    # Finding flux for each element
    al_area = area_calculate(filtered_energy_bins, corrected_counts, al_peak_range)
    si_area = area_calculate(filtered_energy_bins, corrected_counts, si_peak_range)
    mg_area = area_calculate(filtered_energy_bins, corrected_counts, mg_peak_range)
    ca_area = area_calculate(filtered_energy_bins, corrected_counts, ca_peak_range)

    mn_peak_range, cr_peak_range, ti_peak_range, fe_k_peak_range, fe_l_peak_range = (5.85, 5.95), (5.36, 5.46), (4.46, 4.56), (6.35, 6.45), (0.68, 0.75)
    if flare == "high solar flare":
        mn_max_energy, mn_max_count = find_max_in_range(filtered_energy_bins, corrected_counts, mn_peak_range)
        cr_max_energy, cr_max_count = find_max_in_range(filtered_energy_bins, corrected_counts, cr_peak_range)
        ti_max_energy, ti_max_count = find_max_in_range(filtered_energy_bins, corrected_counts, ti_peak_range)
        fe_k_max_energy, fe_k_max_count = find_max_in_range(filtered_energy_bins, corrected_counts, fe_k_peak_range)
        fe_l_max_energy, fe_l_max_count = find_max_in_range(filtered_energy_bins, corrected_counts, fe_l_peak_range)

        mn_area = area_calculate(filtered_energy_bins, corrected_counts, mn_peak_range)
        cr_area = area_calculate(filtered_energy_bins, corrected_counts, cr_peak_range)
        ti_area = area_calculate(filtered_energy_bins, corrected_counts, ti_peak_range)
        fe_k_area = area_calculate(filtered_energy_bins, corrected_counts, fe_k_peak_range)
        fe_l_area = area_calculate(filtered_energy_bins, corrected_counts, fe_l_peak_range)
    else:
        mn_max_energy, mn_max_count = 0,0
        cr_max_energy, cr_max_count = 0,0
        ti_max_energy, ti_max_count = 0,0
        fe_k_max_energy, fe_k_max_count = 0,0
        fe_l_max_energy, fe_l_max_count = 0,0
        mn_area = 0
        cr_area = 0
        ti_area = 0
        fe_k_area = 0
        fe_l_area = 0

    # Filtering out data which don't have counts in energy range of silicon
    if si_area == 0:
        print("si area 0")
        return 1
    total_area = al_area + si_area + mg_area + ca_area + mn_area + cr_area + ti_area + fe_k_area + fe_l_area

    #Calculating the relative abundance of each element in percentage
    al_percent, si_percent, mg_percent, ca_percent, mn_percent, cr_percent, ti_percent, fe_k_percent, fe_l_percent = al_area / total_area * 100, si_area / total_area * 100, mg_area / total_area * 100, ca_area / total_area * 100, mn_area / total_area * 100, cr_area / total_area * 100, ti_area / total_area * 100, fe_k_area / total_area * 100, fe_l_area / total_area * 100

    #Calculating elemental ratios
    al_si_ratio, mg_si_ratio, ca_si_ratio, mn_si_ratio, cr_si_ratio, ti_si_ratio, fe_k_si_ratio, fe_l_si_ratio = al_area / si_area, mg_area / si_area, ca_area / si_area, mn_area / si_area, cr_area / si_area, ti_area / si_area, fe_k_area / si_area, fe_l_area / si_area
    
    values = [file, lat_top_left, lat_top_right, lat_bottom_right, lat_bottom_left, long_top_left, long_top_right, long_bottom_right, long_bottom_left, st_date, end_date, st_timestamp, end_timestamp, angle, side, flare, geotail, al_max_count, si_max_count, mg_max_count, ca_max_count, mn_max_count, cr_max_count, ti_max_count, fe_k_max_count, fe_l_max_count,al_max_energy,si_max_energy, mg_max_energy, ca_max_energy, mn_max_energy, cr_max_energy, ti_max_energy, fe_k_max_energy, fe_l_max_energy, al_area, si_area, mg_area, ca_area, mn_area, cr_area, ti_area, fe_k_area, fe_l_area, al_percent, si_percent, mg_percent, ca_percent, mn_percent, cr_percent, ti_percent, fe_k_percent, fe_l_percent,al_si_ratio, mg_si_ratio, ca_si_ratio, mn_si_ratio, cr_si_ratio, ti_si_ratio, fe_k_si_ratio, fe_l_si_ratio]

    ntd = [0, 9, 10, 11, 12, 14, 15, 16]
    for i in range(len(values)):
        if i not in ntd:
            values[i] = float(values[i])
        else :
            values[i] = str(values[i])

    print("values returned for this")
    return values


def insert(file_path, passwrd, datab, prt):
    values = output(file_path, passwrd, datab, prt)

    if values == 1:
        return False
    column_names = ['file_name', 'u_l_lat', 'u_r_lat', 'b_r_lat', 'b_l_lat', 'u_l_lon', 'u_r_lon', 'b_r_lon', 'b_l_lon', 'start_date', 'end_date', 'start_time', 'end_time', 'angle', 'side', 'flare', 'geotail', 'al_max_counts', 'si_max_counts', 'mg_max_counts','ca_max_counts', 'mn_max_counts', 'cr_max_counts', 'ti_max_counts', 'fe_k_max_counts', 'fe_l_max_counts', 'al_energy','si_energy','mg_energy','ca_energy', 'mn_energy', 'cr_energy', 'ti_energy', 'fe_k_energy', 'fe_l_energy', 'al_area', 'si_area', 'mg_area','ca_area', 'mn_area', 'cr_area', 'ti_area', 'fe_k_area', 'fe_l_area','al_abundance_relative','si_abundance_relative', 'mg_abundance_relative', 'ca_abundance_relative', 'mn_abundance_relative', 'cr_abundance_relative', 'ti_abundance_relative', 'fe_k_abundance_relative', 'fe_l_abundance_relative', 'al/si', 'mg/si', 'ca/si', 'mn/si', 'cr/si', 'ti/si', 'fe_k/si', 'fe_l/si']       # Column names

    connection2 = mysql.connector.connect(
        host="localhost",
        user="root",
        password=passwrd,
        database=datab,
        port = prt
    )

    cursor2 = connection2.cursor()

    column_string = ", ".join([f"`{col}`" for col in column_names])
    placeholder_string = ", ".join(["%s"] * len(values))  # Create a string of "%s" placeholders

    insert_query = f"INSERT INTO catalogue ({column_string}) VALUES ({placeholder_string})"

    # Execute the query
    print("inserting")
    cursor2.execute(insert_query, values)

    connection2.commit()
    print("inserted")
    cursor2.close()
    connection2.close()

    lat_sum = lon_sum = 0
    for i in range(1, 5):
        lat_sum += values[i]
        lon_sum += values[i+4]

    cent_lat = lat_sum/4
    cent_lon = lon_sum/4
    to_give = [cent_lat, cent_lon, values[-8], values[-7], values[-6], values[-5], values[-4], values[-3], values[-2], values[-1]]

    print("generating grid data")
    griding_kriging_automation.update_ratios_in_mysql(to_give, passwrd, datab, prt)

    return True

def update(file_path, passwrd, datab, prt):
    values = output(file_path, passwrd, datab, prt)

    column_names = ['flare', 'geotail', 'al_max_counts', 'si_max_counts', 'mg_max_counts','ca_max_counts', 'mn_max_counts', 'cr_max_counts', 'ti_max_counts', 'fe_k_max_counts', 'fe_l_max_counts', 'al_energy','si_energy','mg_energy','ca_energy', 'mn_energy', 'cr_energy', 'ti_energy', 'fe_k_energy', 'fe_l_energy', 'al_area', 'si_area', 'mg_area','ca_area', 'mn_area', 'cr_area', 'ti_area', 'fe_k_area', 'fe_l_area','al_abundance_relative','si_abundance_relative', 'mg_abundance_relative', 'ca_abundance_relative', 'mn_abundance_relative', 'cr_abundance_relative', 'ti_abundance_relative', 'fe_k_abundance_relative', 'fe_l_abundance_relative', 'al/si', 'mg/si', 'ca/si', 'mn/si', 'cr/si', 'ti/si', 'fe_k/si', 'fe_l/si']       # Column names

    connection2 = mysql.connector.connect(
        host="localhost",
        user="root",
        password=passwrd,
        database=datab,
        port = prt
    )

    cursor2 = connection2.cursor()

    column_string = ", ".join([f"`{col}`" for col in column_names])
    placeholder_string = ", ".join(["%s"] * len(values))  # Create a string of "%s" placeholders

    update_query = f"UPDATE catalogue SET {', '.join(f'`{col}` = %s' for col in column_names)} WHERE `file_name` = %s"

    # Execute the query
    cursor2.execute(update_query, values[15:] + [values[0]])

    # Commit the transaction
    connection2.commit()

    cursor2.close()
    connection2.close()

    lat_sum = lon_sum = 0
    for i in range(1, 5):
        lat_sum += values[i]
        lon_sum += values[i+4]

    cent_lat = lat_sum/4
    cent_lon = lon_sum/4
    to_give = [cent_lat, cent_lon, values[-8], values[-7], values[-6], values[-5], values[-4], values[-3], values[-2], values[-1]]

    print("updating grid data")
    griding_kriging_automation.update_ratios_in_mysql(to_give, passwrd, datab, prt)