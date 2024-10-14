import os
import pandas as pd
import numpy as np
import sys
try:
    sys.path.append("/FitData")
except:
    sys.path.append(".")
#print('Updated sys.path:', sys.path)
# Function to extract the star name from the file name
def extract_star_name(filename):
    return filename.split('-')[0]
def extract_star_id(filename):
    return filename.split('kplr')[1].lstrip('0')
# Load the CSV file containing star names
csv_file = 'pande_teff_logg_vmax.csv'  # Update with the actual filename
stars_df = pd.read_csv(csv_file)
#stars_df = stars_df[stars_df['PRot'] < 35]
#print(star_prod)
# List all directories containing 'public_Q9_long'
def list_directories_with_keyword(directory, keyword):
    folders = []
    for root, dirs, files in os.walk(directory):
        for dir in dirs:
            if keyword in dir:
                folders.append(os.path.join(root, dir))
    return folders

# List all directories containing 'public_Q9_long' within directories specified in sys.path
folders = []
for directory in sys.path:
    folders.extend(list_directories_with_keyword(directory, 'public_Q9_long'))
#print("folders",folders)
match_star=[]
# Loop through each directory
for folder in folders:
    print(folder)
    folder_path = folder
    
    # List all FITS files in the directory
    fits_files = [file for file in os.listdir(folder_path) if file.endswith('.fits')]
    
    # Loop through each FITS file
    for fits_file in fits_files:
        star_name = extract_star_name(fits_file)
        #print(star_name)
        star_name= extract_star_id(star_name)
        #print(star_name)
        # Check if the star name exists in the CSV file
        if int(star_name) in stars_df['KIC'].values:
            #print(star_name)
            #print(f"Star '{star_name}' found in folder '{folder}' and exists in the CSV file.")
            row_with_kic =stars_df[stars_df['KIC'] == int(star_name)]
            # if row_with_kic['Flag'].iloc[0] not in ["SM1","SM2","BQR"]:
            #     match_star.append(star_name)

            # if row_with_kic["Binary"].iloc[0]==0:
            #     match_star.append(star_name)
            match_star.append(star_name)
        else:
            #print(f"Star '{star_name}' found in folder '{folder}' but does not exist in the CSV file.")
            pass
print("Length of the stars that matches:\n",len(match_star))

match_star_array = np.array(match_star)
file_name = 'match_star_array_q9_logg_teff_complete.npy'

# Check if the file exists and delete it if it does
if os.path.exists(file_name):
    os.remove(file_name)

# Now save the NumPy array to a file
np.save(file_name, match_star_array)
