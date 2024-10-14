import sys
import os
import time
start_time = time.time()
sys.path.append('/home/shahriyar/Desktop/Star&Beyond/modules')
import numpy as np
import modules
import torch
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

print("CUDA Available: ",torch.cuda.is_available())
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
has_mps=torch.backends.mps.is_built()
device = "mps" if has_mps else "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device)
modules.mutils.device_status(device)

# PATHS ========================================================================
"""define paths and make new directory for run"""
#RUN = "with_limitation_intepolate_without_limitation"

RUN = "Q9_noisy"
CODE_DIR = "hyperparameter"
HOME_DIR = "home"

DATA_PATH = "/data"
SAVE_PATH = "/models"

SAMPLE_NAME = "/"+"mcquillan"
SLABEL = "prot_27"
LABEL = "/"+SLABEL+"/"
DTYPE = "/"+"best"+"/"

BASELINE = int(SLABEL[-2:])#14 ,27 ,62 or 97
print(BASELINE)
#CADENCE = int(SLABEL[-5:-3])
#print(CADENCE)

paths = {}
paths['home_dir'] = HOME_DIR
paths['code_dir'] = CODE_DIR
paths['data_dir'] = HOME_DIR+DATA_PATH
paths['run_dir'] = HOME_DIR+SAVE_PATH+"/"

if BASELINE != 97:
    LABEL = LABEL[:-4]
#LABEL = LABEL[:-7]

if not os.path.exists(paths['run_dir']+'%s' % RUN):
    os.makedirs(paths['run_dir']+'%s' % RUN)


paths['run_dir'] += "%s/" % RUN
print(paths)


# LOGS =========================================================================
data_log = modules.mutils.create_log(paths['run_dir'], "data_%s" % SLABEL)
timenow = time.time()-start_time
print('[INIT] run time: %.3f s' % timenow, file=data_log)


# LOAD AND PREPARE DATA ========================================================
# get baseline indices
baseline_data = np.load(paths['code_dir']+'/baselines_idx.npy')
baseline_lenidx = np.where(baseline_data[0,:] == BASELINE)[0]
baseline_len = int(baseline_data[1,:][baseline_lenidx])
print("baseline_len",baseline_len)
# load data
data = np.load(paths['data_dir']+'/fluxes/%s/x_values.npy' \
            % RUN)[:, 0:baseline_len]#[:,::cadence_len]
print("data",data.shape)

label = np.load(paths['data_dir']+'/fluxes/%s/y_values.npy' \
            % RUN)
print("label",label.shape)

stds = np.load(paths['data_dir']+'/fluxes/%s/std_values.npy' \
            % RUN)


# train/val/test split
trainidx, validx, testidx = modules.prepare_data.data_split(len(label), data_log)#return the indices, so we have to use data, to have the values

X_train, y_train = modules.prepare_data.scale_data(data[trainidx], label[trainidx],
                                        'train', paths['run_dir'])
X_val, y_val = modules.prepare_data.scale_data(data[validx], label[validx],
                                    'val', paths['run_dir'])
X_test, y_test = modules.prepare_data.scale_data(data[testidx], label[testidx],
                                      'test', paths['run_dir'])
n_train, n_val, n_test = len(y_train), len(y_val), len(y_test)

# get standard deviations
stds_train = modules.prepare_data.scale_stds(stds[trainidx])#Computes the base-10 logarithm of each element in the stds array.
stds_val = modules.prepare_data.scale_stds(stds[validx])
stds_test = modules.prepare_data.scale_stds(stds[testidx])

# batch and prepare for torch
BATCH_SIZE = 256

train_loader = modules.prepare_data.data_for_torch(X_train, stds_train, y_train,
                                        BATCH_SIZE, device, data_log)
val_loader = modules.prepare_data.data_for_torch(X_val, stds_val, y_val,
                                      BATCH_SIZE, device, data_log)
test_loader = modules.prepare_data.data_for_torch(X_test, stds_test, y_test,
                                    X_test.shape[1], device, data_log)
val_pred_loader = modules.prepare_data.data_for_torch(X_val, stds_val, y_val,
                                    X_val.shape[1], device, data_log)
#This type of loader might be useful when you want to evaluate the model's performance on the entire validation set at once, 
#especially for tasks like calculating validation loss or making predictions.


# SAVE FOR TRAINING ============================================================
"save what is necessary for main training loop"
if not os.path.exists(paths['run_dir']+'tmp'):
    os.makedirs(paths['run_dir']+'tmp')

np.save(paths['run_dir']+'tmp/paths.npy', paths)# Saves the dictionary paths into a NumPy binary file named 'paths.npy'. This file contains paths relevant to the project.
np.save(paths['run_dir']+'tmp/lens.npy', np.array([n_train, n_val, n_test]))#Saves an array containing the lengths of the training, validation, and test sets into a NumPy binary file named 'lens.npy'. 
#This file stores information about the lengths of different dataset splits.
np.save(paths['run_dir']+'tmp/datashape.npy', data.shape)# Saves the shape of the data array into a NumPy binary file named 'datashape.npy'. 
#This file stores information about the shape of the dataset.
np.save(paths['run_dir']+'tmp/y_test.npy', y_test)# Saves the y_test array (target labels for the test set) into a NumPy binary file named 'y_test.npy'.
np.save(paths['run_dir']+'tmp/y_val.npy', y_val)

torch.save(train_loader, paths['run_dir']+'tmp/'+'train_loader.pth')#Saves the train_loader DataLoader object into a PyTorch binary file named 'train_loader.pth'. 
#This file contains the DataLoader object necessary for loading and batching training data during model training.
torch.save(val_loader, paths['run_dir']+'tmp/'+'val_loader.pth')
torch.save(val_pred_loader, paths['run_dir']+'tmp/'+'val_pred_loader.pth')
torch.save(test_loader, paths['run_dir']+'tmp/'+'test_loader.pth')
