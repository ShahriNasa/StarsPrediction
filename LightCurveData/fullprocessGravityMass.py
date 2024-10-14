from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from scipy import interpolate
from scipy.interpolate import PchipInterpolator
import shutil
import matplotlib.pyplot as plt
import os
import numpy as np
import gc
import pandas as pd
import numpy as np
from astropy.stats import sigma_clip
from astropy.time import Time
from scipy.interpolate import interp1d
from tqdm import tqdm
from astropy.time import TimeDelta
from astropy import units as u
import gc
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import lightkurve as lk
import pywt
import time
import pickle

def zero_missing_values(sap_flux):    
    missing_indices = np.where(sap_flux == -1)[0]    
    sap_flux[missing_indices] = 0
    
    return sap_flux


def interpolate_missing_values(sap_flux):
    # Get indices where sap_flux is not -1 (valid values)
    valid_indices = np.where(sap_flux != -1)[0]
    
    # Get the values that are not -1
    valid_values = sap_flux[valid_indices]
    
    # Get the indices where sap_flux is -1 (invalid/missing values)
    missing_indices = np.where(sap_flux == -1)[0]
    
    # If there are no missing values, just return the sap_flux as is
    if len(missing_indices) == 0:
        return sap_flux

    # Create a 1D cubic interpolator
    interpolator = interpolate.interp1d(valid_indices, valid_values, kind='linear', fill_value="extrapolate")
    
    # Fill missing values (-1) by interpolation
    sap_flux[missing_indices] = interpolator(missing_indices)
    
    return sap_flux


def extract_curvature(light_curve,level, wavelet='db4'):
    """
    Extract the curvature of the light curve by keeping only the approximation (low-frequency) coefficients.
    
    Parameters:
    - light_curve: array, the input light curve data.
    - wavelet: str, the wavelet to use for decomposition.
    - level: int, the number of decomposition levels.
    
    Returns:
    - Curvature of the light curve (low-frequency component).
    """
    # Step 1: Identify missing values (-1) and replace them with a temporary estimate
    missing_indices = np.where(light_curve == -1)[0]

    if len(missing_indices) == 0:
        coeffs = pywt.wavedec(light_curve, wavelet, level=level)
        # Keep only the approximation coefficients (low-frequency part)
        approximation = coeffs[0]  # First element is the approximation (low-frequency) part
        
        # Reconstruct the signal using only the approximation coefficients
        curvature = pywt.waverec([approximation] + [None] * level, wavelet)
        return curvature[:len(light_curve)]  # No missing values, nothing to impute

    # Linear interpolation for initial guess of missing values
    non_missing_indices = np.where(light_curve != -1)[0]
    non_missing_values = light_curve[non_missing_indices]
    
    interpolator = interp1d(non_missing_indices, non_missing_values, kind='linear', fill_value="extrapolate")
    light_curve_filled = light_curve.copy()
    light_curve_filled[missing_indices] = interpolator(missing_indices)

    # Perform wavelet decomposition
    coeffs = pywt.wavedec(light_curve_filled, wavelet, level=level)
    
    # Keep only the approximation coefficients (low-frequency part)
    approximation = coeffs[0]  # First element is the approximation (low-frequency) part
    
    # Reconstruct the signal using only the approximation coefficients
    curvature = pywt.waverec([approximation] + [None] * level, wavelet)
    
    return curvature[:len(light_curve)]  # Ensure the output length matches the input

def detect_dominant_frequency_wavelet(signal, wavelet='db1', noise_threshold_factor=2):
    coeffs = pywt.wavedec(signal, wavelet)
    
    fs = 1/(29.4*60)  # Sampling frequency
    
    # Calculate power for each detail coefficient
    power = []
    frequencies = []
    
    for i, detail in enumerate(coeffs[1:]):  # Skip the approximation (cA), use only detail coefficients (cD)
        detail_power = np.abs(detail) ** 2
        power.extend(detail_power)
        
        # Calculate corresponding frequencies for each detail coefficient
        scale = 2 ** (i + 1)  # Scale corresponds to wavelet level
        freq = fs / scale
        frequencies.extend([freq] * len(detail))  # Repeat frequency for the length of detail coefficient
    
    # Ensure power and frequencies are numpy arrays
    power = np.array(power)
    frequencies = np.array(frequencies)

    # Set a dynamic threshold for noise
    magnitude_threshold = np.mean(power) + noise_threshold_factor * np.std(power)

    # Apply threshold to filter out insignificant power
    significant_indices = power > magnitude_threshold
    significant_frequencies = frequencies[significant_indices]
    significant_powers = power[significant_indices]

    # Convert significant frequencies to periods in days
    periods_in_days = 1 / significant_frequencies / 3600 / 24  # Convert frequency to period in days

    # Filter out periods less than 1 day
    valid_period_indices = periods_in_days >= 35  # Keep only periods >= 1 day
    valid_frequencies = significant_frequencies[valid_period_indices]
    valid_powers = significant_powers[valid_period_indices]

    # If no valid frequencies are found, return None
    if len(valid_frequencies) == 0:
        return 0

    # Find the dominant frequency
    dominant_freq_index = np.argmax(valid_powers)
    dominant_freq = valid_frequencies[dominant_freq_index]

    # Calculate the period in days
    dominant_period_in_days = 1 / dominant_freq / 3600 if dominant_freq != 0 else np.inf

    return dominant_period_in_days


def detect_dominant_frequency(signal,zero_padding_factor=2, sampling_rate=0.002):
    level=3
    hann = np.hanning(len(signal))
    padded_length = len(signal) * zero_padding_factor
    signal_padded = np.pad(signal, (0, padded_length - len(signal)), 'constant')

    Y = np.fft.fft(signal)
    N = int(len(Y) / 2 + 1)
    fa = 1.0/(29.4*60.0) # every 29 minutes
    X = np.linspace(0, fa/2, N, endpoint=True)
    Xp = 1.0/X # in seconds
    Xph= Xp/(60.0*60.0) # in hours
    # FFT magnitude (normalized)
    magnitude = 2.0 * np.abs(Y[:N]) / N

    # Find indices of the 4 largest peaks in the magnitude (excluding the DC component)
    dominant_indices = np.argsort(magnitude[1:])[::-1][:4] + 1  # +1 to skip the DC component

    thresholdHigh = 2 
    thresholdLow = 35 
    # Extract the dominant frequencies and their magnitudes
    dominant_frequencies = X[dominant_indices]
    dominant_periods_in_hours = 1.0 / dominant_frequencies / 3600

    # Find which periods are less than the threshold (120 hours)
    dominant_High_threshold = dominant_periods_in_hours[dominant_periods_in_hours < thresholdHigh]
    dominant_Low_threshold = dominant_periods_in_hours[dominant_periods_in_hours > thresholdLow]


    if dominant_periods_in_hours[0]/24 <thresholdHigh:
        level=1
    elif dominant_periods_in_hours[0]/24 <5:
        level=2
    elif dominant_periods_in_hours[0]/24>thresholdLow:
        level=3
    else:
        level=3
    return level,dominant_periods_in_hours
def extract_curvature_missing(light_curve,Period_day,level, wavelet='db4' ):
    t = np.linspace(0, 29.4*60, len(light_curve))
    if Period_day>35:
        wavelet='db4'
    else:
        wavelet='db3'
    missing_indices = np.where(light_curve == 0)[0]

    if len(missing_indices) == 0:
        # print("len(missing_indices) == 0")
        coeffs = pywt.wavedec(light_curve, wavelet, level=level)
        # Keep only the approximation coefficients (low-frequency part)
        approximation = coeffs[0]  # First element is the approximation (low-frequency) part
        
        # Reconstruct the signal using only the approximation coefficients
        curvature = pywt.waverec([approximation] + [None] * level, wavelet)
        return curvature[:len(light_curve)]  # No missing values, nothing to impute

    # Linear interpolation for initial guess of missing values
    non_missing_indices = np.where(light_curve != 0)[0]
    non_missing_values = light_curve[non_missing_indices]
    interpolator = interp1d(non_missing_indices, non_missing_values, kind='linear', fill_value="extrapolate")
    light_curve_filled = light_curve.copy()
    light_curve_filled[missing_indices] = interpolator(missing_indices)
    
    
    level,dominant_frequencies=detect_dominant_frequency(light_curve_filled)
    
    coeffs = pywt.wavedec(light_curve_filled, wavelet, level=level)
    
    # Keep only the approximation coefficients (low-frequency part)
    approximation = coeffs[0]  # First element is the approximation (low-frequency) part
    
    # Reconstruct the signal using only the approximation coefficients
    curvature = pywt.waverec([approximation] + [None] * level, wavelet)
    
    return curvature[:len(light_curve)]  # Ensure the output length matches the input


def find_largest_valid_interval(sap_flux):
    # Find the largest contiguous interval of valid values (not equal to -1)
    valid_indices = np.where(sap_flux != -1)[0]
    diff = np.diff(valid_indices)
    gaps = np.where(diff > 1)[0]  # Where there is a gap in valid values
    if len(gaps) == 0:
        # If there are no gaps, the entire series is valid
        return 0, len(sap_flux) - 1
    else:
        # Find the largest contiguous valid interval
        max_len = 0
        start, end = 0, 0
        for i in range(len(gaps) + 1):
            if i == 0:
                s, e = valid_indices[0], valid_indices[gaps[i]]
            elif i == len(gaps):
                s, e = valid_indices[gaps[i-1] + 1], valid_indices[-1]
            else:
                s, e = valid_indices[gaps[i-1] + 1], valid_indices[gaps[i]]

            if e - s > max_len:
                max_len = e - s
                start, end = s, e

        return start, end


#number=1500 for prot
def predict_missing_values(star_data, mse_threshold=0.05,number=4000):
    good_stars = {}
    for star_name, data in tqdm(star_data.items(), desc="Processing stars"):
        # print(f"Processing star: {star_name}")
        # data = data.item()
        # Extract sap_flux and find largest valid interval
        sap_flux = data['sap_flux']

        if np.sum(sap_flux == -1)>number:
            continue
                # data = data.item()

        #sap_flux_interpolate =interpolate_missing_values(sap_flux)
        # sap_flux_wave =extract_curvature(sap_flux)
        sap_flux_wave =zero_missing_values(sap_flux)
        """
        plt.figure(figsize=(10, 6))
        plt.plot(sap_flux_copy, label='Original Data', marker='o')
        plt.plot(sap_flux_interpolate, label='Interpolated Data', linestyle='--', marker='x')

        # Set logarithmic scale for y-axis
        plt.yscale('log')

        # Adding labels and legend
        plt.xlabel('Index')
        plt.ylabel('Flux')
        plt.title('Original and Interpolated Data')
        plt.legend()
        plt.show()
        """
        # Update star data with predicted values
        if np.any(sap_flux_wave  == -1):
            print(f"Warning: sap_flux contains -1 values for star: {star_name}")
            print(np.sum(sap_flux_wave == -1))
        data['sap_flux'] = sap_flux_wave
        # print(len(data['sap_flux']))
        good_stars[star_name] = data

    return good_stars

############################################################################################################################
# Load the .npz file
# file_name="Q9_prot_mass.pkl"
file_name="Q9_teff_logg.pkl"
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

# star_data = np.load('star_pq9_complete_data_PDCSAP_FLUX_time_rotation_name.npz', allow_pickle=True)
# star_data = np.load('star_pq14_complete_data_PDCSAP_FLUX_time_rotation_name.npz', allow_pickle=True)

############################################################################################################################



# The .npz file contains arrays as a dictionary
star_data_dict = star_data
# star_data_dict = {file: star_data[file] for file in star_data.files}
# Now pass the dictionary to your function
updated_star_data = predict_missing_values(star_data_dict, mse_threshold=100)


# Save the updated data with good stars
print("Final Length",len(updated_star_data))
# np.savez_compressed('updatedmissZero_star_q9_data.npz', **updated_star_data)

#star_data = star_array.items()
total_items = len(updated_star_data)
batch_size = 1000


###########################################################################################################

output_dir = 'Q9Wave_complete_batches1500'
# output_dir = 'Q14Wave_complete_batches1500'
# Check if the folder exists
if os.path.exists(output_dir):
    # If it exists, delete the folder and its contents
    shutil.rmtree(output_dir)
##########################################################################################################

os.makedirs(output_dir, exist_ok=True)

# Process data in batches
for start_index in range(0, total_items, batch_size):
    gc.collect()
    end_index = min(start_index + batch_size, total_items)
    batch_data = {key: updated_star_data[key] for key in list(updated_star_data.keys())[start_index:end_index]}
    batch_filename = os.path.join(output_dir, f'batch_{start_index}_{end_index}.npy')

    with open(batch_filename, 'wb') as batch_file:
        np.save(batch_file, batch_data)

###############################################################################################################
input_dir = output_dir
output_dir2 = 'Data9wave_Processed_batches_Interpolate_1500'
# output_dir2 = 'Data14wave_Processed_batches_Interpolate_1500'
if os.path.exists(output_dir2):
    # If it exists, delete the folder and its contents
    shutil.rmtree(output_dir2)
##############################################################################################################

os.makedirs(output_dir2, exist_ok=True)
def outlier_graph(original,preprocessed,time):
    time = pd.to_datetime(time)
    # Calculate the differences to identify the outliers
    # Assuming outliers are where original and interpolated data differ significantly
    outliers = np.abs(original - preprocessed) > 0  # Adjust threshold as needed
    # Plot figure with improved visibility
    plt.figure(figsize=(12, 8))

    # Plot the original data (normal points)
    plt.plot(time,original, 
            label='Original Data (Non-Outliers)', marker='o', markersize=4, color='blue', linestyle='', alpha=0.8)

    # Plot the interpolated data after removing outliers
    plt.plot(time, preprocessed, 
            label='Interpolated Data (After Removing Outliers)', marker='.', markersize=6, color='red', linestyle='', alpha=0.4)

    # Plot the outliers with a different color and marker
    plt.plot(time[outliers], original[outliers], 
            label='Outliers (Original Data)', marker='x', markersize=8, color='green', linestyle='', alpha=1.0)

    # Set logarithmic scale for y-axis
    plt.yscale('log')

    # Adding gridlines
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Adding labels and legend
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Flux (Log Scale)', fontsize=14)
    plt.title('Comparison of Original and Interpolated Data with Outliers Highlighted', fontsize=16)

    # Adding a legend with improved placement
    plt.legend(loc='upper right', fontsize=12)
    plt.xticks(ticks=time[outliers], labels=[t.strftime('%Y-%m-%d %H:%M') for t in time[outliers]], rotation=45, ha='right')

    # Optimizing layout
    plt.tight_layout()

    # Show the plot
    plt.show()

def sigma_clip_windowed(data, window_size=45, sigma_threshold=3):
    clipped_data = data.copy()  
    for i in range(0, len(data), window_size):
        window = data[i:i + window_size]
        mean = np.mean(window)
        std = np.std(window)
        outliers = np.abs(window - mean) > sigma_threshold * std
        window[outliers] = np.nan
        clipped_data[i:i + window_size] = window
    
    return clipped_data

def normalizing(clipped_flux):
    avg_flux = np.mean(clipped_flux)
    delta_flux = clipped_flux - avg_flux
    relative_flux = delta_flux / avg_flux
    mean_flux = np.mean(relative_flux)
    std_flux = np.std(relative_flux)

    normalized_flux = (relative_flux - mean_flux) / std_flux

    return normalized_flux,std_flux

def get_window_size(Period_day):
    # Check boundary conditions
    if Period_day < 5:
        return 45
    elif Period_day > 40:
        return 50
    else:
        # Perform linear interpolation between (5, 5) and (30, 50)
        window_size = 45 + ((Period_day - 5) / (40 - 5)) * (50 - 45)
        return round(window_size)
# Process data in batches
n=0
for filename in os.listdir(input_dir):
    if filename.endswith('.npy'):
        batch_filename = os.path.join(input_dir, filename)
        with open(batch_filename, 'rb') as batch_file:
            batch_data = np.load(batch_file, allow_pickle=True).item()
        batch_filename_save = os.path.join(output_dir2, f'batch_clean_{filename[:-4]}.npy')

    star_ready={}
    keys_to_delete = []
    with tqdm(total=len(batch_data), desc="Processing Batch") as pbar:
        nnn=0
        bbb=0
        sss=0
        fff=0
        period_day_list=[]
        rotation_day_list=[]
        period_day_list_s=[]
        rotation_day_list_s=[]
        for star_id, data in batch_data.items():
            window_size = 50
            data=data            
            if len(data['time']) < 4600:
                n+=1
                print(f"Number bellow 4768: {n}")
                print(len(data['time']))
            if len(data['time'])< 1322:
                keys_to_delete.append(star_id)
                pbar.update(1)
                print(f"Number bellow 1322")
                continue
            time_values, sap_flux = data['time'], data['sap_flux'].copy()
            
            #clipped_flux = []
            # clipped_time = []
            clipped_time = data['time'].copy()
            j = 0
            missing_indices = np.where(sap_flux == 0)[0]
            val_indices = np.where(sap_flux != 0)[0]
            Light_wave=sap_flux.copy()
            missing_indices = np.where(np.isnan(Light_wave) | (Light_wave == 0))[0]
            Light_wave[missing_indices] =0
            non_missing_indices = np.where(Light_wave != 0)[0]
            non_missing_values = Light_wave[non_missing_indices]
            
            interpolator = interp1d(non_missing_indices, non_missing_values, kind='linear', fill_value="extrapolate")
            
            Light_wave[missing_indices] = interpolator(missing_indices)
            level,dominant_frequencies=detect_dominant_frequency(Light_wave)
            Period_day=dominant_frequencies[0]/24
            coeffs_low = pywt.wavedec(Light_wave, "coif5", level=4)
            approximation = coeffs_low[0]  
            # Reconstruct the signal using only the approximation coefficients
            curvature_low = pywt.waverec([approximation] + [None] * 4, "coif5")
            level_low,dominant_frequencies_low=detect_dominant_frequency(curvature_low)
            Period_day_low=dominant_frequencies_low[0]/24
            # Period_day_low=detect_dominant_frequency_wavelet(Light_wave)/24
            # print("Period_day",Period_day)
            # print("Period_day_low",Period_day_low)
            # window_size=int(Period_day)
            # if Period_day<5:
            #     window_size=5
            # elif Period_day>30:
            #     window_size=50
            window_size = get_window_size(Period_day)
            clipped_flux = sigma_clip_windowed(sap_flux[val_indices], window_size=45, sigma_threshold=3)
            sap_flux[val_indices]=clipped_flux
            # Get the values that are not -1
            valid_indices = np.where(~np.isnan(sap_flux) & (sap_flux != 0))[0]
            #valid_indices = np.where(~np.isnan(clipped_flux ))[0]
            # print("Sum",np.isnan(clipped_flux ).sum())
            valid_values = sap_flux[valid_indices]
            #interpolator = interp1d(valid_indices, valid_values, kind='cubic', fill_value="extrapolate")
            missing_indices = np.where(np.isnan(sap_flux) | (sap_flux == 0))[0]
            sap_flux[missing_indices] =0
            if Period_day>0 and Period_day<70 :  #prot 5,25
            # if data['rotation']<5 or data['rotation']>35:
                wave_flux=extract_curvature_missing(sap_flux,Period_day,level)
                normalized_flux,std_flux=normalizing(wave_flux)  
            else:
                normalized_flux,std_flux=normalizing(sap_flux)
                # interpolator = interp1d(valid_indices, normalized_flux[valid_indices], kind='linear', fill_value="extrapolate")
                interpolator = PchipInterpolator(valid_indices, normalized_flux[valid_indices], extrapolate=True)
                normalized_flux[missing_indices] = interpolator(missing_indices)
            if  Period_day > 25:
                period_day_list.append(star_id)
                nnn+=1
            # if  data['rotation']>35:
            #     rotation_day_list.append(star_id)
            #     bbb+=1
            # if dominant_frequencies[0]/24 < 5:
            #     period_day_list_s.append(star_id)
            #     sss+=1
            # if  data['rotation']<5:
            #     rotation_day_list_s.append(star_id)
            #     fff+=1
            # if data['rotation']>35:
            #     print("here: ",data['rotation'],Period_day)
            #sap_flux_wave =extract_curvature(normalized_flux)
            # interpolator = interp1d(valid_indices, normalized_flux[valid_indices], kind='cubic', fill_value="extrapolate")
            # normalized_flux[missing_indices] = interpolator(missing_indices)

            if len(normalized_flux) < 4600:
                n+=1
                # print(f"{len(normalized_flux)} Number bellow 4768: {n}")
            if star_id not in star_ready:
                star_ready[star_id] = {} 
                star_ready[star_id]['sap_flux'] = np.array(normalized_flux)
                star_ready[star_id]['time'] = np.array(clipped_time)
                star_ready[star_id]['standard_dev'] = np.array(std_flux)
                star_ready[star_id]['logg'] = np.array(data["logg"])
                star_ready[star_id]['teff'] = np.array(data["teff"])
            else:
                print("here")
                star_ready[star_id]['sap_flux'] = np.array(normalized_flux)
                star_ready[star_id]['time'] = np.array(clipped_time)
                star_ready[star_id]['standard_dev'] = np.array(std_flux)
                star_ready[star_id]['logg'] = np.array(data["logg"])
                star_ready[star_id]['teff'] = np.array(data["teff"])

            # if star_id not in star_ready:
            #     star_ready[star_id] = {} 
            #     star_ready[star_id]['sap_flux'] = np.array(normalized_flux)
            #     star_ready[star_id]['time'] = np.array(clipped_time)
            #     star_ready[star_id]['standard_dev'] = np.array(std_flux)
            #     star_ready[star_id]['rotation'] = np.array(data["rotation"])
            #     star_ready[star_id]['mass'] = np.array(data["mass"])
            # else:
            #     print("here")
            #     star_ready[star_id]['sap_flux'] = np.array(normalized_flux)
            #     star_ready[star_id]['time'] = np.array(clipped_time)
            #     star_ready[star_id]['standard_dev'] = np.array(std_flux)
            #     star_ready[star_id]['rotation'] = np.array(data["rotation"])
            #     star_ready[star_id]['mass'] = np.array(data["mass"])
            # outlier_graph(relative_flux_o,relative_flux,data['time'])
            pbar.update(1)
    # common_elements = set(period_day_list) & set(rotation_day_list)
    # common_elements_s = set(period_day_list_s) & set(rotation_day_list_s)
    # # Count the number of common elements
    # num_common_elements = len(common_elements)
    # num_common_elements_s = len(common_elements_s)
    # # Display the result
    # print(f"Number of common members above 35: {num_common_elements}")
    # print(f"Number of common members bellow 5: {num_common_elements_s}")
    # print("Length To Delete",len(keys_to_delete))
    # print("Length of Period >35: ",nnn)
    # print("Length of rotation>35: ",bbb)
    # print("Length of Period <5: ",sss)
    # print("Length of rotation<5: ",fff)

    for key in keys_to_delete:
            del star_ready[key]
    with open(batch_filename_save, 'wb') as batch_file:
        np.save(batch_file, star_ready)
    time.sleep(2)

#######################################################################################################################
output_folder_name="DataQ9Wave_ready_1500"
# output_folder_name="DataQ14Wave_ready_1500"
if os.path.exists(output_folder_name):
    # If it exists, delete the folder and its contents
    shutil.rmtree(output_folder_name)
########################################################################################################################


os.makedirs(output_folder_name, exist_ok=True)
batch_filename_save_x = os.path.join(output_folder_name, 'x_values.npy')
batch_filename_save_y = os.path.join(output_folder_name, 'y_values.npy')
batch_filename_save_std = os.path.join(output_folder_name, 'std_values.npy')
folder_path = output_dir2

if folder_path:
    x_values = []
    y_values = []
    std_values = []
    
    npy_files = [filename for filename in os.listdir(folder_path) if filename.endswith('.npy')]

    # Iterate over each file in the selected folder
    for filename in tqdm(npy_files, desc="Processing Files"):
        filepath = os.path.join(folder_path, filename)
        data = np.load(filepath, allow_pickle=True).item()  # Load data from the file
        for star_id, star_data in data.items():

            flux = star_data['sap_flux']
            # rotation = star_data['rotation']  
            # mass = star_data['mass'] 
            teff = star_data['teff'] 
            standard_deviation = star_data['standard_dev']  # Assuming 'standard_deviation' key exists in your data
            
            # Append values to the respective lists
            # if len(flux)<4768:
            #     print(len(flux))
            x_values.append(flux)
            # y_values.append(rotation)

            y_values.append(teff)
            std_values.append(standard_deviation)
            #print(flux.shape)
        
    # Convert lists to numpy arrays
    x_values = np.array(x_values)
    y_values = np.array(y_values)
    std_values = np.array(std_values)
    print(x_values.shape)
    # Save x, y, and standard deviation arrays to .npy files
    np.save(batch_filename_save_x, x_values)
    np.save(batch_filename_save_y, y_values)
    np.save(batch_filename_save_std, std_values)
else:
    print("No folder selected.")

# Close the tkinter root window
if os.path.exists(output_dir):
    # If it exists, delete the folder and its contents
    shutil.rmtree(output_dir)
if os.path.exists(output_dir2):
    # If it exists, delete the folder and its contents
    shutil.rmtree(output_dir2)