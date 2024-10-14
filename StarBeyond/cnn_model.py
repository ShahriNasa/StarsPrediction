import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize  # Adjust transforms as needed
import torch.optim as optim
from tqdm import tqdm 
import modules
import os
import numpy as np
import time
import os.path as path
import re

start_time = time.time()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
modules.mutils.device_status(device)
torch.cuda.memory_allocated()
# Load your custom CNN model
# READ IN DATA AND PATHS =======================================================
HOME_DIR = "home/models/less_35_q5"
HOME_DIR_MODEL="home/models/less_35"
RUN ="dnull_sequence"
RUN ="dnull_sequence"
#DIR = os.path.join(HOME_DIR)
LABEL = "prot_27"
HYPER = "27"
paths = np.load(HOME_DIR+'/tmp/paths.npy', allow_pickle=True).item()
paths_model=np.load(HOME_DIR_MODEL+'/tmp/paths.npy', allow_pickle=True).item()
lens = np.load(paths['run_dir']+'tmp/lens.npy')
n_train, n_val = lens[0], lens[1]
ts_len = np.load(paths['run_dir']+'tmp/datashape.npy')[1]	
param_file = 'hyperparams%s.npy' %HYPER
best_dir =  path.abspath(path.join(os.getcwd(),
                    paths_model['run_dir']))+'/best/%s/' %LABEL
for filename in os.listdir(best_dir):
	if filename.endswith('.png'):
		pid = int(re.search(r'\d+', filename).group())
		break

train_loader = torch.load(paths['run_dir']+'tmp/'+'train_loader.pth')
val_loader = torch.load(paths['run_dir']+'tmp/'+'val_loader.pth')
param_dict = np.load(paths['code_dir']+'/hyperparams/'+param_file, allow_pickle=True)[pid]#here we pick the specific hyperparameter for our next model
paths['transfer'] = paths['run_dir'] + 'transfer_%s_%s/' %(pid, LABEL)
if not os.path.exists(paths['transfer']):
	    os.makedirs(paths['transfer'])
time_log = modules.mutils.create_log(paths['transfer'], "time%s_%s" %(pid, LABEL))
optim_log = modules.mutils.create_log(paths['transfer'], "optim%s_%s" %(pid, LABEL))
model_log = modules.mutils.create_log(paths['transfer'], "model%s_%s" %(pid, LABEL))
performance_log = modules.mutils.create_log(paths['transfer'],
	                            "performance%s_%s" %(pid, LABEL))

pretrained_model=torch.load(best_dir+'model.pt')
print(pretrained_model.keys())
print(pretrained_model['conv1.weight'])
#for param_tensor in pretrained_model:
    #print(param_tensor, "\t", pretrained_model[param_tensor])
#print("Pre train model",pretrained_model.parameters())
print("****"*10)
paths['save_dir'] = paths['transfer'] + 'save%s/' % pid
if not os.path.exists(paths['save_dir']):
	    os.makedirs(paths['save_dir'])

net = modules.model.CNN(num_in=ts_len,
		        log=model_log,
		        kernel1=int(param_dict['KERNEL_1']),
		        kernel2=int(param_dict['KERNEL_2']),
		        padding1=int(param_dict['PADDING_1']),
		        padding2=int(param_dict['PADDING_2']),
		        stride1=int(param_dict['STRIDE_1']),
		        stride2=int(param_dict['STRIDE_2']),
		        dropout=float(param_dict['DROPOUT']))

train_log = modules.mutils.create_log(paths['save_dir'], "train_%s" % LABEL)


#net.load_state_dict(pretrained_model)
net.conv1.weight.data=pretrained_model['conv1.weight']
net.conv2.weight.data=pretrained_model['conv2.weight']
net.conv2.bias.data = pretrained_model['conv2.bias']
net.conv1.bias.data = pretrained_model['conv1.bias']
net.bn1.weight.data = pretrained_model['bn1.weight']
net.bn2.weight.data = pretrained_model['bn2.weight']
 
net.conv1.weight.requires_grad = False
net.conv1.bias.requires_grad = False
net.conv2.weight.requires_grad = False
net.conv2.bias.requires_grad = False
net.bn1.weight.requires_grad=False
net.bn2.weight.requires_grad=False

N_EPOCHS = 200
N_STOP = 50
TOL = .01
net.to(device)
optimizer = torch.optim.AdamW(net.parameters(),
	                     lr=float(param_dict['LR']),
	                     weight_decay=float(param_dict['WD']),
	                     eps=float(param_dict['EPS']))

#print("Net model",net.state_dict())
print("****"*10)
"""
for param_tensor in net.state_dict():
    print(param_tensor, "\t", net.state_dict()[param_tensor].size())
#net.load_state_dict(pretrained_model)
for param_tensor in net.state_dict():
    print(param_tensor, "\t", net.state_dict()[param_tensor])
"""
loss_func = torch.nn.MSELoss()
train_log = modules.mutils.create_log(paths['save_dir'], "train_%s" % LABEL)
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
print(model.conv1.weight.data)

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
# SAVE =========================================================================
"save model performance"
print('PID: %s, r2: %.3f, bias: %.3f, rms: %.3f' %(pid,r2,bias,rms))
print('PID: %s, r2: %.3f, bias: %.3f, rms: %.3f' %(pid,r2,bias,rms),
		file=performance_log)
timenow = time.time()-start_time
print('[FINISH %s] run time: %.3f s' % (pid, timenow))
print('[FINISH %s] run time: %.3f s' % (pid, timenow), file=time_log)
print("*****\n","Finished......... ")

