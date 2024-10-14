import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import modules
from sklearn.metrics import r2_score, mean_squared_error

#### Loading the complete model

device = torch.device("mps" if torch.backends.mps.is_built() else "cuda" if torch.cuda.is_available() else "cpu")
HOME_DIR = "home/models/Q9_db"
RUN ="Q9_db"
LABEL = "prot_27"
HYPER = "27"
folder="/Best_Grid/"
n="pid2"
pid="2"
paths = np.load(HOME_DIR+'/tmp/paths.npy', allow_pickle=True).item()
model = torch.load(HOME_DIR+folder+n+'/model_complete.pth')
model.to(device)  # Send model to device (e.g., GPU or CPU)

#####Loading New Data
for name, param in model.named_parameters():
    print(name)

HOME_DIR = "home/models/DataQ14Wave_ready_1500"
RUN ="DataQ14Wave_ready_1500"
#DIR = os.path.join(HOME_DIR)
LABEL = "prot_27"
HYPER = "27"
paths = np.load(HOME_DIR+'/tmp/paths.npy', allow_pickle=True).item()
paths['run_dir']="home/models/DataQ14Wave_ready_1500/"
# RUN="resultComplete_1"Q9_db
RUN="Transfer_Best"
train_loader = torch.load(paths['run_dir']+'tmp/'+'train_loader.pth')
val_loader = torch.load(paths['run_dir']+'tmp/'+'val_loader.pth')
############################################
#Saving Directory

HOME_DIR = "home/models/Q9_db"
RUN ="Q9_db"
#DIR = os.path.join(HOME_DIR)
LABEL = "prot_27"
HYPER = "27"
paths = np.load(HOME_DIR+'/tmp/paths.npy', allow_pickle=True).item()
paths['run_dir']="home/models/Q9_db/"
# RUN="resultComplete_1"
RUN="Transfer_DataQ14Wave_ready_1500"
lens = np.load(paths['run_dir']+'tmp/lens.npy')
n_train, n_val = lens[0], lens[1]
ts_len = np.load(paths['run_dir']+'tmp/datashape.npy')[1]
if not os.path.exists(paths['run_dir']+'%s' % RUN):
    os.makedirs(paths['run_dir']+'%s' % RUN)
paths['run_dir_mine']=paths['run_dir']+'%s' % RUN+'/'
paths['save_dir'] = paths['run_dir_mine'] + 'pid%s/' % pid
if not os.path.exists(paths['save_dir']):
    os.makedirs(paths['save_dir'])
# Freeze the weights of conv1 and conv2 layers
def freeze_lstm_cells(lstm_layer):
    for name, param in lstm_layer.named_parameters():
        if 'i2h' in name:# or 'h2h' in name:  # Freezing only input-to-hidden and hidden-to-hidden layers
            param.requires_grad = False
def freeze_layer(layer):
    for param in layer.parameters():
        param.requires_grad = False
freeze_layer(model.conv1)
# # freeze_layer(model.conv3)
freeze_layer(model.conv2)
# # freeze_layer(model.bn3)
freeze_layer(model.bn2)
freeze_layer(model.bn1)
# freeze_lstm_cells(model.lstm)
# freeze_layer(model.linear2)
# freeze_layer(model.linear1)
# Optionally, you could also modify the fully connected layers 
# if the output dimensions of the new task are different
# Example: If you want to modify the last fully connected layer
# (Here we assume the new task still requires output of size 1, adjust as needed)
# model.predict = nn.Linear(hidden3 or hidden2, new_output_size)

# Print the model to verify the layers are correctly frozen
print(model)

# Define the optimizer, ensuring it only updates the unfrozen parameters
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),  # Only update unfrozen layers
    lr=0.0001,  # Specify the new learning rate for fine-tuning
    weight_decay=0.0001
)

# Define loss function
loss_func = nn.MSELoss()  # Modify if your new task requires a different loss



def transfer_learning_train(model, device, n_epochs, train_loader, val_loader, save_path, log, n_stop=20, tol=0.01):
    early_stopped = False
    no_improve = 0
    current_val_loss = float('inf')
    losses = np.zeros((2, n_epochs))
    r2_scores = np.zeros(n_epochs)
    rmse_scores = np.zeros(n_epochs)
    best_model_wts = None  # Store best model weights based on validation loss

    for epoch in range(n_epochs):
        model.train()  # Set model to training mode

        # Train on the new data
        running_train_loss = 0.0
        for x, s, y in train_loader:
            optimizer.zero_grad()
            x, s, y = x.to(device), s.to(device), y.to(device)

            predictions = model(x, s)
            loss = loss_func(predictions, y)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)
        losses[1, epoch] = avg_train_loss

        # Validation loop
        model.eval()  # Set model to evaluation mode
        running_val_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for xval, sval, yval in val_loader:
                xval, sval, yval = xval.to(device), sval.to(device), yval.to(device)
                val_predictions = model(xval, sval)
                val_loss = loss_func(val_predictions, yval)
                running_val_loss += val_loss.item()

                # Collect predictions and true values for R² and RMSE calculation
                all_preds.append(val_predictions.cpu().numpy())
                all_labels.append(yval.cpu().numpy())

        avg_val_loss = running_val_loss / len(val_loader)
        losses[0, epoch] = avg_val_loss

        # Flatten predictions and labels for metric calculation
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        # Calculate R² and RMSE
        r2 = r2_score(all_labels, all_preds)
        rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
        r2_scores[epoch] = r2
        rmse_scores[epoch] = rmse

        # Early stopping check
        if avg_val_loss < current_val_loss:
            if current_val_loss - avg_val_loss > tol * avg_val_loss:
                no_improve = 0
            current_val_loss = avg_val_loss
            best_model_wts = model.state_dict()  # Save the best model weights
            torch.save(model.state_dict(), save_path + 'transfer_model.pt')  # Save the best model
        else:
            no_improve += 1

        if no_improve >= n_stop:
            early_stopped = True
            print('=========EARLY STOPPING=========')
            break

        # Print training, validation losses, R², and RMSE at each epoch
        print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {avg_train_loss:.5f}, "
              f"Val Loss: {avg_val_loss:.5f}, R²: {r2:.5f}, RMSE: {rmse:.5f}")

    # Load the best model weights (based on validation loss)
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)

    # Final evaluation on the validation set for R² and RMSE
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for xval, sval, yval in val_loader:
            xval, sval, yval = xval.to(device), sval.to(device), yval.to(device)
            val_predictions = model(xval, sval)
            all_preds.append(val_predictions.cpu().numpy())
            all_labels.append(yval.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    final_r2 = r2_score(all_labels, all_preds)
    final_rmse = np.sqrt(mean_squared_error(all_labels, all_preds))

    print(f"\nFinal R²: {final_r2:.5f}")
    print(f"Final RMSE: {final_rmse:.5f}")

    # Save the final model if not early stopped
    if not early_stopped:
        torch.save(model.state_dict(), save_path + 'transfer_model_final.pt')

    # Call plotting function for losses and R² scores
    plot_metrics(losses, r2_scores, save_path)

    # Plot predicted vs. true values
    metrics = {
        'r2': final_r2,
        'bias': np.mean(all_preds - all_labels),
        'rms': final_rmse
    }
    plot_pred_true(save_path, 'Predicted vs True', pid, all_preds, all_labels, metrics)

# Function to plot losses and R² scores
def plot_metrics(losses, r2_scores, save_path):
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(16, 4))

    # Plot training and validation loss
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:blue')
    ax1.semilogy(losses[0, :], label='Validation Loss', color='tab:blue')
    ax1.semilogy(losses[1, :], label='Training Loss', color='tab:cyan')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Create a second y-axis for R² scores
    ax2 = ax1.twinx()
    ax2.set_ylabel('R² Score', color='tab:red')
    ax2.plot(r2_scores, label='R² Score', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    fig.tight_layout()  # Ensure everything fits well

    plt.title('Training/Validation Loss and R² Score over Epochs')
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    
    # Save the plot
    plt.savefig(save_path + 'loss_r2_plot.png')
    # plt.show()

# Plot predicted vs true values
def plot_pred_true(save_path, label, pid, y_pred, y_true, metrics):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = plt.axes()
    plt.title(r'label: %s, pid: %s' % (label, pid))

    ax.scatter(y_true, y_pred, s=1, c='k')
    ax.plot(y_true, y_true)
    ax.set_ylabel('Predicted')
    ax.set_xlabel('True')
    ax.text(0.05, 0.93, r'r²: %s, Δ: %s, RMS: %s' % (np.round(metrics['r2'], 2),
                                                      np.round(metrics['bias'], 2),
                                                      np.round(metrics['rms'], 2)),
            fontsize=10, ha='left', transform=ax.transAxes)
    ax.text(0.5,0.02, r'true: $\mu$=%s, $\sigma$=%s, \
            pred: $\mu$=%s, $\sigma$=%s' % (np.round(np.mean(y_true),3),
                                            np.round(np.std(y_true),3),
                                            np.round(np.mean(y_pred),3),
                                            np.round(np.std(y_pred),3)),
            fontsize=10, ha='left', transform=ax.transAxes)
    plt.tight_layout()
    plt.savefig(save_path + 'true_v_pred%s.png' % pid)
    # plt.show()

train_log = modules.mutils.create_log(paths['save_dir'], "train_%s" % LABEL)
# Call the transfer learning training function
transfer_learning_train(model, device, n_epochs=300,  # Adjust based on your task
                        train_loader=train_loader,  # New dataset loader
                        val_loader=val_loader,  # New validation loader
                        save_path=paths['save_dir'],  # Save path
                        log=train_log,  # Log file
                        n_stop=20, tol=0.01)
