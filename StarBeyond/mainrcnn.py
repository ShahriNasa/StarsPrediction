import sys

import modules.rcnnfinal
print(sys.version)
import os
import time
from tqdm import tqdm
import pickle

start_time = time.time()
import numpy as np
sys.path.append('/home/shahriyar/Desktop/Star&Beyond/modules')
import modules

import optuna
from torch.optim import AdamW
import torch
import torch.nn as nn
import torch.nn.functional as F
from optuna.pruners import MedianPruner
torch.cuda.empty_cache()

print("CUDA Available: ",torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
modules.mutils.device_status(device)



# READ IN DATA AND PATHS =======================================================
HOME_DIR = "home/models/teff6_window_45_45_0_70"
RUN ="teff6_window_45_45_0_70"
#DIR = os.path.join(HOME_DIR)
LABEL = "prot_27"
HYPER = "27"
pid = 141
paths = np.load(HOME_DIR+'/tmp/paths.npy', allow_pickle=True).item()
print("paths['run_dir']",paths['run_dir'])
paths['run_dir']="home/models/teff6_window_45_45_0_70/"
print(paths)
train_loader = torch.load(paths['run_dir']+'tmp/'+'train_loader.pth')
val_loader = torch.load(paths['run_dir']+'tmp/'+'val_loader.pth')

lens = np.load(paths['run_dir']+'tmp/lens.npy')
n_train, n_val = lens[0], lens[1]
ts_len = np.load(paths['run_dir']+'tmp/datashape.npy')[1]
def reset_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
#def check_validity(num_in, kernel1, kernel2, stride1, stride2, padding1, padding2, poolsize1, poolsize2):
#def check_validity(num_in, kernel1, kernel2, kernel3, stride1, stride2, stride3, padding1, padding2, padding3, poolsize1, poolsize2, poolsize3=2):

def check_validity(num_in, kernel1, kernel2, kernel3, stride1, stride2, stride3, padding1, padding2, padding3, poolsize1, poolsize2, poolsize3):
    # Check if kernel sizes are valid with respect to the input sizes at each stage

    # After first convolution
    num_out = ((num_in + 2 * padding1 - (kernel1 - 1) - 1) / stride1) + 1
    if  str(num_out)[-1] != '0':
        return False
    
    num_out = num_out / poolsize1
    if num_out < kernel2 or str(num_out)[-1] != '0':  # num_out should also be positive
        return False

    # After second convolution
    num_out = ((num_out + 2 * padding2 - (kernel2 - 1) - 1) / stride2) + 1
    if  str(num_out)[-1] != '0':
        return False

    num_out = num_out / poolsize2

    
    if num_out < kernel3 or str(num_out)[-1] != '0':
        return False

    # After third convolution
    num_out = ((num_out + 2 * padding3 - (kernel3 - 1) - 1) / stride3) + 1
    if  str(num_out)[-1] != '0':
        return False

    num_out = num_out / poolsize3
    if  str(num_out)[-1] != '0':
        return False
    return True


# def check_validity(num_in, kernel1, kernel2,kernel3, stride1, stride2,stride3, padding1, padding2,padding3, poolsize1=4, poolsize2=4, poolsize3=2):
#     # Same validity checking logic as before
#     num_out = ((num_in + 2*padding1 - (kernel1 - 1) - 1) / stride1) + 1
#     if str(num_out)[-1] != '0':
#         return False

#     num_out = num_out / poolsize1
#     if str(num_out)[-1] != '0':
#         return False

#     num_out = ((num_out + 2*padding2 - (kernel2 - 1) - 1) / stride2) + 1
#     if str(num_out)[-1] != '0':
#         return False

#     num_out = num_out / poolsize2
#     if str(num_out)[-1] != '0':
#         return False

#     num_out = ((num_out + 2*padding3 - (kernel3 - 1) - 1) / stride3) + 1
#     if str(num_out)[-1] != '0':
#         return False

#     num_out = num_out / poolsize3
#     if str(num_out)[-1] != '0':
#         return False

#     return True

# def Model_Initializer(OUT_CHANNELS_1,OUT_CHANNELS_2,poolsize1,poolsize2, trial, kernel1, kernel2, padding1, padding2, stride1, stride2, dropout, lr, wd, eps):
#     model = modules.model.CNN(OUT_CHANNELS_1=OUT_CHANNELS_1,OUT_CHANNELS_2=OUT_CHANNELS_2,poolsize1=poolsize1,poolsize2=poolsize2,num_in=ts_len, log=None, kernel1=kernel1, kernel2=kernel2,
#                               padding1=padding1, padding2=padding2,
#                               stride1=stride1, stride2=stride2, dropout=dropout)

def Model_Initializer(OUT_CHANNELS_1,OUT_CHANNELS_2,OUT_CHANNELS_3,poolsize1,poolsize2,poolsize3,hidden1,hidden2,hidden3, trial, kernel1, kernel2, kernel3, padding1, padding2, padding3, stride1, stride2, stride3, dropout, lr, wd, eps,rnn_hidden_size,rnn_num_layers):
    model = modules.rcnnfinal.RCNN(OUT_CHANNELS_1=OUT_CHANNELS_1,OUT_CHANNELS_2=OUT_CHANNELS_2,OUT_CHANNELS_3=OUT_CHANNELS_3,poolsize1=poolsize1,poolsize2=poolsize2,poolsize3=poolsize3,hidden1=hidden1,hidden2=hidden2,hidden3=hidden3,num_in=ts_len, log=None, kernel1=kernel1, kernel2=kernel2, kernel3=kernel3,
                              padding1=padding1, padding2=padding2, padding3=padding3,
                              stride1=stride1, stride2=stride2, stride3=stride3, dropout=dropout,rnn_hidden_size=rnn_hidden_size,rnn_num_layers=rnn_num_layers)


    model.to(device)
    
    # Training setup
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd, eps=eps)
    loss_func = torch.nn.MSELoss()
    ProgressBar=tqdm(range(100), desc="Epochs", unit="epoch")
    for epoch in ProgressBar:
        model.train()
        training_loss = 0.0

        for x_batch, s_batch, y_batch in tqdm(train_loader, desc="Training Batches", leave=False):
            x_batch, s_batch, y_batch = x_batch.to(device), s_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(x_batch, s_batch)
            loss = loss_func(output, y_batch)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()

        training_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for x_val, s_val, y_val in tqdm(val_loader, desc="Validation Batches", leave=False):
                x_val, s_val, y_val = x_val.to(device), s_val.to(device), y_val.to(device)
                output = model(x_val, s_val)
                val_loss += loss_func(output, y_val).item()

        val_loss /= len(val_loader)
        
        # Check for divergence
        divergence_ratio = val_loss / training_loss
        ProgressBar.set_description(f"Training (Val Loss: {training_loss:.5f},Train Loss: {val_loss:.5f})")
        # if divergence_ratio > 4:
        #     print(f"Pruning trial due to divergence: val_loss={val_loss}, training_loss={training_loss}, ratio={divergence_ratio:.2f}")
        #     raise optuna.TrialPruned()
        
        # Report validation loss to Optuna
        trial.report(val_loss, epoch)

        # Early stopping
        if trial.should_prune():
            raise optuna.TrialPruned()

    return val_loss


def objective(trial):
    # Suggest hyperparameters for the trial
    # OUT_CHANNELS_1 = trial.suggest_int('OUT_CHANNELS_1', 4, 65,step=4)
    # OUT_CHANNELS_2 = trial.suggest_int('OUT_CHANNELS_2',4, 60,step=2)
    # OUT_CHANNELS_3 = trial.suggest_int('OUT_CHANNELS_3', 0, 60,step=2)
    OUT_CHANNELS_1 = trial.suggest_categorical('OUT_CHANNELS_1', [64, 32,16])
    OUT_CHANNELS_2 = trial.suggest_categorical('OUT_CHANNELS_2', [8,4,2])
    OUT_CHANNELS_3 = trial.suggest_categorical('OUT_CHANNELS_3', [8,4,2])
    poolsize1=trial.suggest_categorical('poolsize1', [4, 2])
    poolsize2=trial.suggest_categorical('poolsize2', [4, 2,1])
    poolsize3=trial.suggest_categorical('poolsize3', [4, 2,1])
    hidden1=trial.suggest_categorical('hidden1', [ 1024,512,256,128])
    hidden2=trial.suggest_categorical('hidden2', [512, 256,128,64])
    hidden3=trial.suggest_categorical('hidden3', [256,128,64,32,0])
    kernel1 = trial.suggest_int('KERNEL_1', 5, 25,step=5)
    kernel2 = trial.suggest_int('KERNEL_2', 25, 35,step=5)
    kernel3 = trial.suggest_int('KERNEL_3', 35, 60,step=5)
    padding1 = trial.suggest_int('PADDING_1', 0, 3,step=1)
    padding2 = trial.suggest_int('PADDING_2', 0, 3,step=1)
    padding3 = trial.suggest_int('PADDING_3', 0, 3,step=1)
    stride1 = trial.suggest_int('STRIDE_1', 1, 3,step=1)
    stride2 = trial.suggest_int('STRIDE_2', 1, 3,step=1)
    stride3 = trial.suggest_int('STRIDE_3', 1, 3,step=1)
    dropout = trial.suggest_categorical('DROPOUT', [0.5,0.4, 0.2,0.3])
    lr = trial.suggest_categorical('LR', [ 1e-4, 1e-5])
    wd = trial.suggest_categorical('WD', [1e-6, 1e-4,])
    eps = trial.suggest_categorical('EPS', [1e-8])
    # rnn_hidden_size=trial.suggest_int('RNNSIZE',2,64,step=2)
    # rnn_num_layers=trial.suggest_int('RNNLAYER',1,6,step=1)
    rnn_hidden_size=trial.suggest_categorical('RNNSIZE',[64,32,16,8])
    rnn_num_layers=trial.suggest_categorical('RNNLAYER',[2,3,4])

    
    # Validate the combination

    if not check_validity(num_in=ts_len, kernel1=kernel1, kernel2=kernel2,kernel3=kernel3,
                          stride1=stride1, stride2=stride2,stride3=stride3,
                          padding1=padding1, padding2=padding2,padding3=padding3,poolsize1=poolsize1,poolsize2=poolsize2,poolsize3=poolsize3):
        raise optuna.TrialPruned()  # Skip this trial if the configuration is invalid

    # Initialize the model
    try:
        print("OUT_CHANNELS_1 ",OUT_CHANNELS_1,"OUT_CHANNELS_2 ",OUT_CHANNELS_2,"OUT_CHANNELS_3 ",OUT_CHANNELS_3,"poolsize1 ",poolsize1,"poolsize2 ",poolsize2,"poolsize3 ",poolsize3)
        print("hidden1 ",hidden1,"hidden2 ",hidden2,"hidden3 ",hidden3,"kernel1 ",kernel1,"kernel2 ",kernel2,"kernel3 ",kernel3)
        print("RNN SIZE ",rnn_hidden_size,"RNN LAYER ",rnn_num_layers)
        print("dropout ",dropout,"lr ",lr,"wd ",wd,"eps ",eps)
        return Model_Initializer(OUT_CHANNELS_1,OUT_CHANNELS_2,OUT_CHANNELS_3,poolsize1,poolsize2,poolsize3,hidden1,hidden2,hidden3,trial,kernel1,kernel2,kernel3,padding1,padding2,padding3,stride1,stride2,stride3,dropout,lr,wd,eps,rnn_hidden_size,rnn_num_layers)
        #return Model_Initializer(OUT_CHANNELS_1,OUT_CHANNELS_2,poolsize1,poolsize2,trial,kernel1,kernel2,padding1,padding2,stride1,stride2,dropout,lr,wd,eps)
    except:
        raise optuna.TrialPruned() 
    finally:
        torch.cuda.empty_cache()

# Run the Optuna Study
# pruner = optuna.pruners.MedianPruner(n_warmup_steps=5, interval_steps=10)
# sampler=optuna.samplers.BruteForceSampler(seed=None)
# sampler=optuna.samplers.TPESampler()
# sampler=optuna.samplers.GPSampler(deterministic_objective=True,n_startup_trials=1000)
sampler=optuna.samplers.NSGAIISampler()
# study = optuna.create_study(direction='minimize', sampler=sampler)
study = optuna.create_study(direction='minimize',sampler=sampler)

study.optimize(objective, n_trials=5000)

# Print the best hyperparameters
print(f"Best hyperparameters: {study.best_trial.params}")
trials_df = study.trials_dataframe()
trials_df = trials_df[trials_df['state'] == 'COMPLETE']

trials_df.to_csv('wavelet_teff6_27.csv', index=False)

print("All trial results saved to optuna_study_results.csv")
with open('optuna_study.pkl', 'wb') as f:
    pickle.dump(study, f)
# Print the best hyperparameters
if device=='cuda':
    torch.cuda.empty_cache()


print("CUDA Available: ",torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
modules.mutils.device_status(device)



# READ IN DATA AND PATHS =======================================================
HOME_DIR = "home/models/teff6_window_45_45_0_70"
RUN ="teff6_window_45_45_0_70"
#DIR = os.path.join(HOME_DIR)
LABEL = "prot_27"
HYPER = "27"
paths = np.load(HOME_DIR+'/tmp/paths.npy', allow_pickle=True).item()
print("paths['run_dir']",paths['run_dir'])
paths['run_dir']="home/models/teff6_window_45_45_0_70/"
print(paths)
train_loader = torch.load(paths['run_dir']+'tmp/'+'train_loader.pth')
val_loader = torch.load(paths['run_dir']+'tmp/'+'val_loader.pth')

lens = np.load(paths['run_dir']+'tmp/lens.npy')
n_train, n_val = lens[0], lens[1]
ts_len = np.load(paths['run_dir']+'tmp/datashape.npy')[1]
pid=111111

time_log = modules.mutils.create_log(paths['run_dir'], "time%s_%s" %(pid, LABEL))
optim_log = modules.mutils.create_log(paths['run_dir'], "optim%s_%s" %(pid, LABEL))
model_log = modules.mutils.create_log(paths['run_dir'], "model%s_%s" %(pid, LABEL))
performance_log = modules.mutils.create_log(paths['run_dir'],
								"performance%s_%s" %(pid, LABEL))


# HYPERPARAMETERS ==============================================================
paths['save_dir'] = paths['run_dir'] + 'pid%s/' % pid

if not os.path.exists(paths['save_dir']):
	os.makedirs(paths['save_dir'])



# DEFINE MODEL =================================================================
net = modules.rcnnfinal.RCNN(num_in=ts_len,
                log=model_log,
                OUT_CHANNELS_1=int(study.best_trial.params['OUT_CHANNELS_1']),
                OUT_CHANNELS_2=int(study.best_trial.params['OUT_CHANNELS_2']),
                OUT_CHANNELS_3=int(study.best_trial.params['OUT_CHANNELS_3']),
                poolsize1=int(study.best_trial.params['poolsize1']),
                poolsize2=int(study.best_trial.params['poolsize2']),
                poolsize3=int(study.best_trial.params['poolsize3']),
                hidden1=int(study.best_trial.params['hidden1']),
                hidden2=int(study.best_trial.params['hidden2']),
                hidden3=int(study.best_trial.params['hidden3']),
                kernel1=int(study.best_trial.params['KERNEL_1']),
                kernel2=int(study.best_trial.params['KERNEL_2']),
                kernel3=int(study.best_trial.params['KERNEL_3']),
                padding1=int(study.best_trial.params['PADDING_1']),
                padding2=int(study.best_trial.params['PADDING_2']),
                padding3=int(study.best_trial.params['PADDING_3']),
                stride1=int(study.best_trial.params['STRIDE_1']),
                stride2=int(study.best_trial.params['STRIDE_2']),
                stride3=int(study.best_trial.params['STRIDE_3']),
                dropout=float(study.best_trial.params['DROPOUT']),
                rnn_hidden_size=int(study.best_trial.params['RNNSIZE']),
                rnn_num_layers=int(study.best_trial.params['RNNLAYER']))

net.to(device)

print(net)
print(net, file=model_log)


# TRAIN ========================================================================
train_log = modules.mutils.create_log(paths['save_dir'], "train_%s" % LABEL)

N_EPOCHS = 100
N_STOP = 5
TOL = .01

optimizer = torch.optim.AdamW(net.parameters(),
						lr=float(study.best_trial.params['LR']),
						weight_decay=float(study.best_trial.params['WD']),
						eps=float(study.best_trial.params['EPS']))

loss_func = torch.nn.MSELoss()
print("Training")
modules.train.training(model=net,
		device=device,
		n_stop=N_STOP,
		tol=TOL,
		optimizer=optimizer,
		loss_func=loss_func,
		n_epochs=N_EPOCHS,
		train_loader=train_loader,
		n_train=n_train,
		val_loader=val_loader,
		n_val=n_val,
		save_path=paths['save_dir'],
		log=train_log, pid=pid)


print("Training is finished")

# EVALUATE =====================================================================
model = net
model.load_state_dict(torch.load(paths['save_dir']+'model.pt'))
print("Evaluation")
# make predictions
val_pred_loader = torch.load(paths['run_dir']+'tmp/'+'val_pred_loader.pth')
y_val_pred = modules.evaluate.predictions(device, model, val_pred_loader)
test_loader = torch.load(paths['run_dir']+'tmp/'+'test_loader.pth')
y_test_pred = modules.evaluate.predictions(device, model, test_loader)

# transform back to original data scaling
y_test_pred = modules.evaluate.inverse_scale(paths['run_dir'], 'test', y_test_pred)
y_test = np.load(paths['run_dir']+'tmp/y_test.npy')
y_test = modules.evaluate.inverse_scale(paths['run_dir'], 'test', y_test)

y_val_pred = modules.evaluate.inverse_scale(paths['run_dir'], 'val', y_val_pred)
y_val = np.load(paths['run_dir']+'tmp/y_val.npy')
y_val = modules.evaluate.inverse_scale(paths['run_dir'], 'val', y_val)

# save for plotting later
np.save(paths['save_dir']+'y_test_pred.npy', y_test_pred)
np.save(paths['save_dir']+'y_test_true.npy', y_test)
np.save(paths['save_dir']+'y_val_pred.npy', y_val_pred)
np.save(paths['save_dir']+'y_val_true.npy', y_val)
print("computing evaluation metrics")
# compute evaluation metrics
r2 = modules.evaluate.r2(y_val_pred, y_val)
bias = modules.evaluate.bias(y_val_pred, y_val)
rms = modules.evaluate.rms(y_val_pred, y_val)

metrics = {'r2':r2,
	'bias':bias,
	'rms':rms}
print("Saving Plots")
# make and save plots
modules.evaluate.plot_pred_true(paths['save_dir'], LABEL, pid,
					y_val_pred, y_val, metrics)

print("Saving ")
# SAVE =========================================================================
"save model performance"
print('PID: %s, r2: %.3f, bias: %.3f, rms: %.3f' %(pid,r2,bias,rms))
print('PID: %s, r2: %.3f, bias: %.3f, rms: %.3f' %(pid,r2,bias,rms),
	file=performance_log)

timenow = time.time()-start_time
print('[FINISH %s] run time: %.3f s' % (pid, timenow))
print('[FINISH %s] run time: %.3f s' % (pid, timenow), file=time_log)
print("*****\n","Finished......... ")
