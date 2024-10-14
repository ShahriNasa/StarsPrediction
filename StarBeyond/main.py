import sys
print(sys.version)
import os
import time
from tqdm import tqdm

start_time = time.time()
import numpy as np
sys.path.append('/home/shahriyar/Desktop/Star&Beyond/modules')
import modules


import torch
print("CUDA Available: ",torch.cuda.is_available())
has_mps=torch.backends.mps.is_built()
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = "mps" if has_mps else "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device)
modules.mutils.device_status(device)


# READ IN DATA AND PATHS =======================================================
HOME_DIR = "home/models/less_35"
RUN ="less_35"
#DIR = os.path.join(HOME_DIR)
LABEL = "prot_27"
HYPER = "27"
pid = 141
paths = np.load(HOME_DIR+'/tmp/paths.npy', allow_pickle=True).item()
print("paths['run_dir']",paths['run_dir'])
print(paths)
train_loader = torch.load(paths['run_dir']+'tmp/'+'train_loader.pth')
val_loader = torch.load(paths['run_dir']+'tmp/'+'val_loader.pth')

lens = np.load(paths['run_dir']+'tmp/lens.npy')
n_train, n_val = lens[0], lens[1]
ts_len = np.load(paths['run_dir']+'tmp/datashape.npy')[1]
param_file = 'hyperparams%s.npy' %HYPER

for pid in range(0,144):
#for pid in [0,1,2,3,59,69,74,101,128,129,130,131,140,141,142]:
	time_log = modules.mutils.create_log(paths['run_dir'], "time%s_%s" %(pid, LABEL))
	optim_log = modules.mutils.create_log(paths['run_dir'], "optim%s_%s" %(pid, LABEL))
	model_log = modules.mutils.create_log(paths['run_dir'], "model%s_%s" %(pid, LABEL))
	performance_log = modules.mutils.create_log(paths['run_dir'],
		                            "performance%s_%s" %(pid, LABEL))


	# HYPERPARAMETERS ==============================================================
	paths['save_dir'] = paths['run_dir'] + 'pid%s/' % pid

	if not os.path.exists(paths['save_dir']):
	    os.makedirs(paths['save_dir'])

	param_dict = np.load(paths['code_dir']+'/hyperparams/'+param_file, allow_pickle=True)[pid]#here we pick the specific hyperparameter for our next model with the PID value
	print('[HYPERPARAM %s] params: %s' % (pid, param_dict), file=optim_log)


	# DEFINE MODEL =================================================================
	net = modules.model.CNN(num_in=ts_len,
		        log=model_log,
		        kernel1=int(param_dict['KERNEL_1']),
		        kernel2=int(param_dict['KERNEL_2']),
		        padding1=int(param_dict['PADDING_1']),
		        padding2=int(param_dict['PADDING_2']),
		        stride1=int(param_dict['STRIDE_1']),
		        stride2=int(param_dict['STRIDE_2']),
		        dropout=float(param_dict['DROPOUT']))

	net.to(device)

	print(net)
	print(net, file=model_log)


	# TRAIN ========================================================================
	train_log = modules.mutils.create_log(paths['save_dir'], "train_%s" % LABEL)

	N_EPOCHS = 100
	N_STOP = 50
	TOL = .01

	optimizer = torch.optim.AdamW(net.parameters(),
		                     lr=float(param_dict['LR']),
		                     weight_decay=float(param_dict['WD']),
		                     eps=float(param_dict['EPS']))

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
