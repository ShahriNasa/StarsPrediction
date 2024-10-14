
import sys
print(sys.version)
import os
import time
from tqdm import tqdm
import pickle
import modules.rcnnfinal

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
import pandas as pd
# hyperparams_df = pd.read_csv("wavelet_teff2.csv")
# hyperparams_df = pd.read_csv('wavelet_teff6_27.csv')
hyperparams_df = pd.read_csv("Best_Grid.csv")

torch.cuda.empty_cache()



print("CUDA Available: ",torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
modules.mutils.device_status(device)


# READ IN DATA AND PATHS =======================================================
HOME_DIR = "home/models/Q9_noisy"
RUN ="Q9_noisy"
#DIR = os.path.join(HOME_DIR)
LABEL = "prot_27"
HYPER = "27"
paths = np.load(HOME_DIR+'/tmp/paths.npy', allow_pickle=True).item()
print("paths['run_dir']",paths['run_dir'])
paths['run_dir']="home/models/Q9_noisy/"
# RUN="resultComplete_1"
RUN="27"
train_loader = torch.load(paths['run_dir']+'tmp/'+'train_loader.pth')
val_loader = torch.load(paths['run_dir']+'tmp/'+'val_loader.pth')

lens = np.load(paths['run_dir']+'tmp/lens.npy')
n_train, n_val = lens[0], lens[1]
ts_len = np.load(paths['run_dir']+'tmp/datashape.npy')[1]
if not os.path.exists(paths['run_dir']+'%s' % RUN):
    os.makedirs(paths['run_dir']+'%s' % RUN)
paths['run_dir_mine']=paths['run_dir']+'%s' % RUN+'/'

for index, row in hyperparams_df.iterrows():
    torch.cuda.empty_cache()
    pid=int(row['number'])

    time_log = modules.mutils.create_log(paths['run_dir_mine'], "time%s_%s" %(pid, LABEL))
    optim_log = modules.mutils.create_log(paths['run_dir_mine'], "optim%s_%s" %(pid, LABEL))
    model_log = modules.mutils.create_log(paths['run_dir_mine'], "model%s_%s" %(pid, LABEL))
    performance_log = modules.mutils.create_log(paths['run_dir_mine'],
                                    "performance%s_%s" %(pid, LABEL))


    # HYPERPARAMETERS ==============================================================
    paths['save_dir'] = paths['run_dir_mine'] + 'pid%s/' % pid

    if not os.path.exists(paths['save_dir']):
        os.makedirs(paths['save_dir'])
    net = modules.rcnnfinal.RCNN(num_in=ts_len,
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
                rnn_hidden_size=int(row['params_RNNSIZE']),
                rnn_num_layers=int(row['params_RNNLAYER']))

    net.to(device)

    print(net)
    print(net, file=model_log)


    # TRAIN ========================================================================
    train_log = modules.mutils.create_log(paths['save_dir'], "train_%s" % LABEL)
    arch_log = modules.mutils.create_log_path(paths['save_dir'], "network_architecture")
    base_path = os.path.dirname(__file__)
    arch_log=os.path.join(base_path,arch_log)
    with open(str(arch_log), "w") as arch_log:
        print("CNN Channels:\n", "CHANNELS 1:", row['params_OUT_CHANNELS_1'], "CHANNELS 2:", row['params_OUT_CHANNELS_2'], "CHANNELS 3:", row['params_OUT_CHANNELS_3'], "\n", file=arch_log)
        print("CNN Kernel Size:\n", "KERNEL 1:", row['params_KERNEL_1'], "KERNEL 2:", row['params_KERNEL_2'], "KERNEL 3:", row['params_KERNEL_3'], "\n", file=arch_log)
        print("CNN Padding Size:\n", "PADDING 1:", row['params_PADDING_1'], "PADDING 2:", row['params_PADDING_2'], "PADDING 3:", row['params_PADDING_3'], "\n", file=arch_log)
        print("CNN Stride Size:\n", "STRIDE 1:", row['params_STRIDE_1'], "STRIDE 2:", row['params_STRIDE_2'], "STRIDE 3:", row['params_STRIDE_3'], "\n", file=arch_log)
        print("Pooling Size:\n", "Pooling 1:", row['params_poolsize1'], "Pooling 2:", row['params_poolsize2'], "Pooling 3:", row['params_poolsize3'], "\n", file=arch_log)
        print("LSTM Number of Layers:\n", row['params_RNNLAYER'], "\n", file=arch_log)
        print("LSTM Hidden Size:\n", row['params_RNNSIZE'], "\n", file=arch_log)
        print("Fully Connected Architecture:\n", row['params_hidden1'], row['params_hidden2'], row['params_hidden3'], "\n", file=arch_log)
        print("Fully Connected Dropout:\n", row['params_DROPOUT'], "\n", file=arch_log)

    N_EPOCHS = 800
    N_STOP = 5
    TOL = .01
    optimizer = torch.optim.AdamW(net.parameters(),
						lr=float(row['params_LR']),
						weight_decay=float(row['params_WD']),
						eps=float(row['params_EPS']))

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
    torch.cuda.empty_cache()
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
