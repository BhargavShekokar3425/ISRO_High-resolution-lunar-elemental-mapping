import pandas as pd
import numpy as np
import os
import argparse
from astropy.io import fits

def to_native_endian(array, dtype=np.float64):
    if array.dtype.byteorder not in ('=', '|'):  # Check if the array is non-native endian
        return array.astype(dtype)
    return array


def output(take_date):

    file_path = f"background_files/{take_date}.csv"
    bg_data = pd.read_csv(file_path)

    total_counts = bg_data["0"]

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

    # Filter energy range between 0.5 and 10 keV
    energy_mask = (energy_bins >= 0.5) & (energy_bins <= 10.0)
    filtered_counts = total_counts[energy_mask]
    filtered_effective_area = effective_area[energy_mask]
    filtered_energy_bins = energy_bins[energy_mask]

    # Correct for effective area
    corrected_counts = filtered_counts / filtered_effective_area

    mg_energy_range = (1.20, 1.30)
    mg_mask = (filtered_energy_bins >= mg_energy_range[0]) & (filtered_energy_bins <= mg_energy_range[1])
    mg_background_counts = corrected_counts[mg_mask]
    mg_mean = np.mean(mg_background_counts)
    mg_std = np.std(mg_background_counts)

    al_energy_range = (1.43, 1.53)
    al_mask = (filtered_energy_bins >= al_energy_range[0]) & (filtered_energy_bins <= al_energy_range[1])
    al_background_counts = corrected_counts[al_mask]
    al_mean = np.mean(al_background_counts)
    al_std = np.std(al_background_counts)
    
    si_energy_range = (1.68, 1.78)
    si_mask = (filtered_energy_bins >= si_energy_range[0]) & (filtered_energy_bins <= si_energy_range[1])
    si_background_counts = corrected_counts[si_mask]
    si_mean = np.mean(si_background_counts)
    si_std = np.std(si_background_counts)

    ca_energy_range = (3.64, 3.74)
    ca_mask = (filtered_energy_bins >= ca_energy_range[0]) & (filtered_energy_bins <= ca_energy_range[1])
    ca_background_counts = corrected_counts[ca_mask]
    ca_mean = np.mean(ca_background_counts)
    ca_std = np.std(ca_background_counts)

    mn_energy_range = (5.85, 5.95)
    mn_mask = (filtered_energy_bins >= mn_energy_range[0]) & (filtered_energy_bins <= mn_energy_range[1])
    mn_background_counts = corrected_counts[mn_mask]
    mn_mean = np.mean(mn_background_counts)
    mn_std = np.std(mn_background_counts)

    cr_energy_range = (5.36, 5.46)
    cr_mask = (filtered_energy_bins >= cr_energy_range[0]) & (filtered_energy_bins <= cr_energy_range[1])
    cr_background_counts = corrected_counts[cr_mask]
    cr_mean = np.mean(cr_background_counts)
    cr_std = np.std(cr_background_counts)

    ti_energy_range = (4.46, 4.56)
    ti_mask = (filtered_energy_bins >= ti_energy_range[0]) & (filtered_energy_bins <= ti_energy_range[1])
    ti_background_counts = corrected_counts[ti_mask]
    ti_mean = np.mean(ti_background_counts)
    ti_std = np.std(ti_background_counts)

    fe_k_energy_range = (6.35, 6.45)
    fe_k_mask = (filtered_energy_bins >= fe_k_energy_range[0]) & (filtered_energy_bins <= fe_k_energy_range[1])
    fe_k_background_counts = corrected_counts[fe_k_mask]
    fe_k_mean = np.mean(fe_k_background_counts)
    fe_k_std = np.std(fe_k_background_counts)
    
    fe_l_energy_range = (0.68, 0.75)
    fe_l_mask = (filtered_energy_bins >= fe_l_energy_range[0]) & (filtered_energy_bins <= fe_l_energy_range[1])
    fe_l_background_counts = corrected_counts[fe_l_mask]
    fe_l_mean = np.mean(fe_l_background_counts)
    fe_l_std = np.std(fe_l_background_counts)
    
    return [take_date, al_mean,mg_mean,si_mean,ca_mean,mn_mean,cr_mean,ti_mean,fe_k_mean,fe_l_mean,al_std,mg_std,si_std,ca_std,mn_std,cr_std,ti_std,fe_k_std,fe_l_std ]