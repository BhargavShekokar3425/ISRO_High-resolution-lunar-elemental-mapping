## README

**Overview**

This repository contains various Python scripts, data files, and resources for processing, analyzing, and visualizing data related to Inter IIT ISRO PS Final Evaluation. This repository belongs to Team 73. This document provides a detailed explanation of the directory structure, the purpose of each file, and instructions for setting up and running the project.

**Directory Structure**

**Root Directory: Codes**

The root directory contains the primary Python scripts and datasets required for the project. Below is a description of each file and subdirectory:

**Python Files**

* **background_mean_sigma_counts.py:** Script to calculate mean and sigma counts for background data.
* **best_ratios.ipynb:** Jupyter Notebook analyzing the best elemental ratios from the provided data.
* **catalogue_generator.py:** Generates a catalog from the input data files for further processing.
* **class_arf_v1.arf and class_rmf_v1.rmf:** Auxiliary files for handling ARF (Auxiliary Response File) and RMF (Response Matrix File) data.
* **dash_visualizer.py:** Provides a visualization dashboard for data exploration and analysis.
* **file_analyzer.py:** Analyzes input files and prepares them for processing.
* **geotail.csv:** A dataset containing geotail data.
* **griding_kriging_automation.py:** Automates the process of gridding and kriging for spatial data.
* **grid_results_using_clipped_data.csv, high_solar_flare_intervals.csv, kriged_results_using_clipped_catalogue.csv:** Intermediate results and datasets used in processing and analysis.
* **lroc_color_poles_1k.jpg:** A visual image file related to lunar data.
* **sql_utility.py:** Contains SQL setup and utility functions required for processing data.
* **subpixel_resolution_multi.py:** Handles subpixel resolution adjustments for datasets.
* **xsm_data_generator.py:** Generates data from XSM (X-ray Spectrometer) sources.

**Subdirectories**

* **background_files:** Contains CSV files representing background data for analysis.
* **New_Data:** Contains new, unprocessed data.
    * **fits_files:** Empty directory intended for FITS files.
    * **xsm_files:** Empty directory intended for XSM files.
* **processed_data:** Contains processed data files.
    * **fits_files:**
        * **dayside:** Placeholder directory for FITS files from dayside observations.
        * **nightside:** Placeholder directory for FITS files from nightside observations.
    * **xsm_files:** Placeholder directory for processed XSM files.


**Setup and Usage**

**Prerequisites**

* Python: Ensure Python 3.8 or later is installed.
* SQL: Install and configure a SQL database (e.g., MySQL, PostgreSQL).

**Installation Steps**

1. Clone the repository to your local machine.
2. Set up a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate   # On Windows: env\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Configure the SQL database using the `sql_utility.py` script.

**Execution Steps**

1. Run the SQL setup script:
   ```bash
   python sql_utility.py
   ```
2. Process the files using the analyzer:
   ```bash
   python file_analyzer.py
   ```
3. Start the dashboard for visualization:
   ```bash
   python dash_visualizer.py
   ```
4. Open the provided URL in a browser to interact with the dashboard.

**Notes**

* Ensure all dependencies are installed before running the scripts.
* Modify paths in scripts if running in a non-default environment.
* Use the `background_files` directory for any additional background data required.
