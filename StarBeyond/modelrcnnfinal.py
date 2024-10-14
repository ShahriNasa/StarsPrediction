import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
import os
import re
import modules
from modules.rcnnfinal import RCNN
import torch.nn as nn
import pandas as pd
start_time = time.time()
device = torch.device("mps" if torch.backends.mps.is_built() else "cuda" if torch.cuda.is_available() else "cpu")

# Load your custom RCNN model
# READ IN DATA AND PATHS =======================================================
HOME_DIR = "home/models/less_35_q5"
HOME_DIR_MODEL="home/models/original"
LABEL = "pid358"
hyperparams_df = pd.read_csv('optuna_study_original_NSG_Fully.csv')

paths = np.load(HOME_DIR+'/tmp/paths.npy', allow_pickle=True).item()
paths_model=np.load(HOME_DIR_MODEL+'/tmp/paths.npy', allow_pickle=True).item()
paths_model['run_dir']="home/models/original"
lens = np.load(paths['run_dir']+'tmp/lens.npy')
n_train, n_val = lens[0], lens[1]
ts_len = np.load(paths['run_dir']+'tmp/datashape.npy')[1]	
best_dir =  os.path.abspath(os.path.join(os.getcwd(), paths_model['run_dir']))+'/resultoriginal/%s/' %LABEL
for filename in os.listdir(best_dir):
	if filename.endswith('.png'):
		pid = int(re.search(r'\d+', filename).group())
		break

train_loader = torch.load(paths['run_dir']+'tmp/'+'train_loader.pth', map_location=device)
val_loader = torch.load(paths['run_dir']+'tmp/'+'val_loader.pth', map_location=device)
paths['transferfinal'] = paths['run_dir'] + 'transfer_%s_%s/' %(pid, LABEL)
if not os.path.exists(paths['transferfinal']):
	    os.makedirs(paths['transferfinal'])
time_log = open(os.path.join(paths['transferfinal'], "time%s_%s" %(pid, LABEL)), "w")
optim_log = open(os.path.join(paths['transferfinal'], "optim%s_%s" %(pid, LABEL)), "w")
model_log = open(os.path.join(paths['transferfinal'], "model%s_%s" %(pid, LABEL)), "w")
performance_log = open(os.path.join(paths['transferfinal'], "performance%s_%s" %(pid, LABEL)), "w")

pretrained_model = torch.load(best_dir+'model.pt', map_location=device)
print(pretrained_model.keys())
print(pretrained_model['conv3.weight'])
print("****"*10)
paths['save_dir'] = os.path.join(paths['transferfinal'], 'save%s/' % pid)
if not os.path.exists(paths['save_dir']):
	    os.makedirs(paths['save_dir'])
for index, row in hyperparams_df.iterrows():
    if int(row['number'])==358:
        torch.cuda.empty_cache()
        net = RCNN(num_in=ts_len,
                log=model_log,
                OUT_CHANNELS_1=int(row['params_OUT_CHANNELS_1']),
                OUT_CHANNELS_2=int(row['params_OUT_CHANNELS_2']),
                OUT_CHANNELS_3=int(row['params_OUT_CHANNELS_3']),
                poolsize1=int(row['params_poolsize1']),
                poolsize2=int(row['params_poolsize2']),
                poolsize3=int(row['params_poolsize3']),
                hidden1=int(row['params_hidden1']),
                hidden2=int(row['params_hidden2']),
                hidden3=int(row['params_hidden3']),
                kernel1=int(row['params_KERNEL_1']),
                kernel2=int(row['params_KERNEL_2']),
                kernel3=int(row['params_KERNEL_3']),
                padding1=int(row['params_PADDING_1']),
                padding2=int(row['params_PADDING_2']),
                padding3=int(row['params_PADDING_3']),
                stride1=int(row['params_STRIDE_1']),
                stride2=int(row['params_STRIDE_2']),
                stride3=int(row['params_STRIDE_3']),
                dropout=float(row['params_DROPOUT']),
                rnn_hidden_size=128,
                rnn_num_layers=2)


        print(net)
        net.conv1.weight.data = pretrained_model['conv1.weight']
        net.conv2.weight.data = pretrained_model['conv2.weight']
        net.conv2.bias.data = pretrained_model['conv2.bias']
        net.conv1.bias.data = pretrained_model['conv1.bias']
        net.bn1.weight.data = pretrained_model['bn1.weight']
        net.bn2.weight.data = pretrained_model['bn2.weight']
        net.conv3.weight.data = pretrained_model['conv3.weight']
        net.conv3.bias.data = pretrained_model['conv3.bias']
        net.bn3.weight.data = pretrained_model['bn3.weight']

        # net.conv1.weight.requires_grad = False
        # net.conv1.bias.requires_grad = False
        # net.conv2.weight.requires_grad = False
        # net.conv2.bias.requires_grad = False
        # net.bn1.weight.requires_grad = False
        # net.bn2.weight.requires_grad = False
        # net.conv3.weight.requires_grad = False
        # net.conv3.bias.requires_grad = False
        # net.bn3.weight.requires_grad = False


        N_EPOCHS = 800
        N_STOP = 10
        TOL = .01
        net.to(device)
        optimizer = torch.optim.AdamW(net.parameters(),
                                lr=float(row['params_LR']),
                                weight_decay=float(row['params_WD']),
                                eps=float(row['params_EPS']))

        loss_func = nn.MSELoss()
        train_log = open(os.path.join(paths['save_dir'], "train_%s" % LABEL), "w")
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
        val_pred_loader = torch.load(paths['run_dir']+'tmp/'+'val_pred_loader.pth',map_location=device)
        y_val_pred = modules.evaluate.predictions(device, model, val_pred_loader)
        test_loader = torch.load(paths['run_dir']+'tmp/'+'test_loader.pth',map_location=device)
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
