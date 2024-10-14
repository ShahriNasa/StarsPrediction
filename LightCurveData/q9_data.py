import os
import pandas as pd
import numpy as np
import sys
from astropy.timeseries import TimeSeries
import pickle
import astropy.io.fits as fits
from astropy.time import Time

ask_csv=None
ask_npy=None
ask_output=None
ask_csvv=None
####CSV File
while True:
    ask_csvv = str(input("Please Choose: \n1-mcquillan_prot_M.csv\n2-pande_teff_logg_vmax.csv\n3-berger_classification.csv\n"))
    if ask_csvv=="1":
        ask_csv="mcquillan_prot_M.csv"
        break
    elif ask_csvv=="2":
        ask_csv="pande_teff_logg_vmax.csv"
        break
    elif ask_csvv=="3":
        ask_csv="berger_classification.csv"
        break
    else:
        print("Please choose the correct number.")
    

####match star File
while True:
    ask_npy = str(input("Please enter the Quarter (q9 or q5): \n"))

    if ask_npy=="q9" or ask_npy=="Q9" :
        if ask_csvv=="1":
            ask_npy="match_star_array_q9_prot_complete.npy"
            break
        elif ask_csvv=="2":
            ask_npy="match_star_array_q9_logg_teff_complete.npy"
            break
        elif ask_csvv=="3":
            ask_npy="match_star_array_q9_evolve_complete.npy"
            break
    elif ask_npy=="q5" or ask_npy=="Q5" :
        if ask_csvv=="1":
            ask_npy="match_star_array_q5_prot_complete.npy"
            break
        elif ask_csvv=="2":
            ask_npy="match_star_array_q5_logg_teff_complete.npy"
            break
        elif ask_csvv=="3":
            ask_npy="match_star_array_q5_evolve_complete.npy"
            break
    elif ask_npy=="q14" or ask_npy=="Q14" :
        if ask_csvv=="1":
            ask_npy="match_star_array_q14_prot_complete.npy"
            break
        elif ask_csvv=="2":
            ask_npy="match_star_array_q14_logg_teff_complete.npy"
            break
        elif ask_csvv=="3":
            ask_npy="match_star_array_q14_evolve_complete.npy"
            break
    
    else:
        print("We don't have this, select another one....")
####OutPut File Name
while True:
    ask_output = str(input("Please enter the output name (we'll cover the format[extension]): "))

    ask_output=ask_output+".pkl"
    break
try:
    sys.path.append("/media/shahriyar/ShAhRiYaR/LightCurveData")
    sys.path.append("/Volumes/ShAhRiYaR/LightCurveData")
    sys.path.append("/run/user/1000/gvfs/smb-share:server=nasa,share=usbdisk4_volume1/LightCurveData")
    # sys.path.append("/run/user/1000/gvfs/smb-share:server=nasa,share=usbdisk1_volume1/LightCurveData")
except:
    sys.path.append("/media/shahriyar/ShAhRiYaR/LightCurveData")

# csv_file = 'mcquillan_prot_M.csv' 
csv_file = ask_csv
stars_df = pd.read_csv(csv_file)
# Load the match star array
# match_star_array = np.load('match_star_array_q9_prot_complete.npy')

match_star_array = np.load(ask_npy)
# Function to extract the star name from the file name
def extract_star_name(filename):
    return filename.split('-')[0]
def extract_star_id(filename):
    return filename.split('kplr')[1].lstrip('0')


# Function to extract the light curve data and SAP quality flag from a FITS file
def extract_light_curve(fits_file):
    with fits.open(fits_file) as hdul:
        # The light curve data is usually in the first data extension
        data = hdul[1].data
        # columns = data.columns.names
        # print("Available columns:", columns)
        sap_flux = data['PDCSAP_FLUX']
        sap_quality = data['sap_quality']
        time = data['time']
        # Identify the indices where time is NaN
        nan_indices = np.isnan(time)
        
        # Interpolate to fill NaN values based on neighboring time values
        if np.any(nan_indices):  # Check if there are any NaN values
            # Interpolate the NaN times using linear interpolation
            time[nan_indices] = np.interp(
                np.flatnonzero(nan_indices),    # Positions where time is NaN
                np.flatnonzero(~nan_indices),   # Positions where time is not NaN
                time[~nan_indices]              # Corresponding time values where it's not NaN
            )
            bjd_time = time + 2454833
            # Convert BJD to calendar date (ISO format) using astropy
            time_obj = Time(bjd_time, format='jd')  # Convert to Julian Date (JD)
            time = time_obj.iso 
            time = np.asarray(time)
            sap_flux[nan_indices] = -1
            sap_quality[nan_indices] = 1
        
        return np.asarray(sap_flux), sap_quality, time


# List all directories containing 'public_Q9_long'
def list_directories_with_keyword(directory, keyword):
    folders = []
    for root, dirs, files in os.walk(directory):
        for dir in dirs:
            if keyword in dir:
                folders.append(os.path.join(root, dir))
    return folders
def save_star_data(star_data, file_name="star_data_incremental.pkl"):
    # Append mode: save star data in small chunks
    with open(file_name, 'ab') as f:  # Open in append-binary mode
        pickle.dump(star_data, f, protocol=pickle.HIGHEST_PROTOCOL)

def process_and_save_fits_files(fits_files, folder_path, stars_df, match_star_array,filename,features=ask_csvv, batch_size=50):
    star_data_batch = {}
    for i, fits_file in enumerate(fits_files):
        star_name = extract_star_name(fits_file)
        star_name = extract_star_id(star_name)

        if star_name in match_star_array:
            sap_flux, sap_quality, time = extract_light_curve(os.path.join(folder_path, fits_file))
            selected_star = stars_df[stars_df["KIC"] == int(star_name)]
            
            sap_flux = np.where(sap_quality > 0, -1, sap_flux)
            if features=='3':
                evolve = selected_star["Evolved"].iloc[0]
                teff = selected_star["Teff"].iloc[0]
                if star_name not in star_data_batch:           
                    star_data_batch[star_name] = {
                        'sap_flux': sap_flux,
                        'time': time,
                        'evolve': evolve,
                        'teff': teff
                    }
            elif features=='2':
                logg = selected_star["logg"].iloc[0]
                teff = selected_star["T_eff"].iloc[0]
                if star_name not in star_data_batch:           
                    star_data_batch[star_name] = {
                        'sap_flux': sap_flux,
                        'time': time,
                        'logg': logg,
                        'teff': teff
                    }
            elif features=='1':
                rotation = selected_star["PRot"].iloc[0]
                mass = selected_star["Mass"].iloc[0]
                if star_name not in star_data_batch: 
                    star_data_batch[star_name] = {
                        'sap_flux': sap_flux,
                        'time': time,
                        'rotation': rotation,
                        'mass': mass
                    }
            

        # Save the batch of star data every `batch_size` FITS files
        if (i + 1) % batch_size == 0 or i == len(fits_files) - 1:
            save_star_data(star_data_batch,file_name=filename)
            star_data_batch.clear()  # Clear the batch from memory

# List all directories containing 'public_Q9_long' within directories specified in sys.path
folders = []
for directory in sys.path:
    folders.extend(list_directories_with_keyword(directory, 'public_Q9_long'))

star_data = {}
file_name=ask_output
if os.path.exists(file_name):
    os.remove(file_name)

for folder in folders:
    print("Fetching From: ",folder)
    fits_files = [file for file in os.listdir(folder) if file.endswith('.fits')]
    process_and_save_fits_files(fits_files, folder, stars_df, match_star_array,filename=ask_output,batch_size=1000)


def load_star_data(file_name=file_name):
    star_data = {}
    with open(file_name, 'rb') as f:
        while True:
            try:
                star_data.update(pickle.load(f))  # Load and update the dictionary
            except EOFError:
                break  # Stop when reaching the end of the file
    return star_data


# Load data later when needed
star_data = load_star_data(file_name=file_name)
print(len(star_data))