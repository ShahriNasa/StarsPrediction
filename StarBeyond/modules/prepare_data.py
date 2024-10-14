import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import torch
from torch.utils.data import TensorDataset, DataLoader




def data_split(len_data, log):
    """
    Split the data three ways into a training, validation, and test set.
        
    Parameters
    ----------
    len_data : int
        Number of stars in sample.
    
    log : _io.TextIOWrapper
        Log file.
        
    Returns
    ----------
    train_idx : array_like, int
        Training set indices.
        
    val_idx : array_like, int
        Validation set indices.
        
    test_idx : array_like, int
        Test set indices.
    """
    idx = np.arange(0, len_data)  # Indices for the entire dataset
    # Xtrainval, Xtest, ytrainval, ytest = train_test_split(idx, idx, test_size=0.20, random_state=23)

    # # Second split: Split the 80% train+validation into 60% train and 20% validation
    # idx = np.arange(0, len(Xtrainval))  # Indices for the 80% train+validation set
    # Xtrain, Xval, ytrain, yval = train_test_split(idx, idx, test_size=0.25, random_state=23)
    # # test_size=0.25 means 25% of the 80% (which is 20% of the total) will be used for validation

    # # Final index assignments
    # train_idx = Xtrainval[Xtrain]  # Indices for the training set (60% of total)
    # val_idx = Xtrainval[Xval]      # Indices for the validation set (20% of total)
    # test_idx = Xtest  


    Xtrainval, Xtest, ytrainval, ytest = train_test_split(idx, idx,
                                    test_size=0.15, random_state=23)#

    idx = np.arange(0, len(Xtrainval))#0.85 percent of the data
    Xtrain, Xval, ytrain, yval = train_test_split(idx, idx,
                                    test_size=0.15, random_state=23)
                                    
    train_idx = Xtrainval[Xtrain]
    val_idx = Xtrainval[Xval]
    test_idx = Xtest
    assert len(train_idx)+len(val_idx)+len(test_idx) == len_data
    
    print('Total number of stars: %s' % len_data, file=log)
    print('Total number of stars: %s'% len_data)
    print('%s train, %s validation, %s test' % (len(train_idx),
                                                len(val_idx),
                                                len(test_idx)), file=log)
    print('%s train, %s validation, %s test' % (len(train_idx),
                                                len(val_idx),
                                                len(test_idx)))
    print(r'%s train, %s validation, %s test' % (len(train_idx)/len_data,
                                        len(val_idx)/len_data,
                                        len(test_idx)/len_data), file=log)
    print(r'%s train, %s validation, %s test' % (len(train_idx)/len_data,
                                        len(val_idx)/len_data,
                                        len(test_idx)/len_data))
                                
    return train_idx, val_idx, test_idx


def scale_data(data, label, set, dir):
    """
    Scale the input time series data, as well as the target stellar properties.
        
    Parameters
    ----------
    data : array_like, floats
        Light curve flux values.
        
    label : array_like, floats
        Stellar property array.
        
    set : string
        Name of dataset (train, val, or test).
        
    dir : string
        Directory where scaler will be saved.
        
    Returns
    ----------
    X_scaled : array_like, floats
        Scaled time series data.
        
    y_scaled : array_like, floats
        Scaled stellar property data.
    """
    X_scaled = StandardScaler().fit_transform(data.T)#This line standardizes the input time series data using the StandardScaler from scikit-learn
    #StandardScaler scales each feature (column) of the input data to have a mean of 0 and a standard deviation of 1.
    y_scaler = MinMaxScaler((0,1)).fit(label.reshape(-1, 1))#MinMaxScaler scales each feature to a given range, in this case, (0,1).
    y_scaled = y_scaler.transform(label.reshape(-1, 1))#label.reshape(-1, 1) reshapes the label array to ensure it is a column vector (necessary for MinMaxScaler).
    #This transformation is essential in machine learning pipelines to ensure that input features and target variables are on similar scales, which can help improve the performance and convergence of 
    #machine learning algorithms. Scaling ensures that no single feature dominates the learning process due to its larger magnitude. It also helps in making the optimization process more stable and 
    #efficient.
    pickle.dump(y_scaler, open(dir+'y_%s_scaler.sav' % set, 'wb'))

    return X_scaled, y_scaled


def scale_stds(stds):
    """
    Scale the light curve standard deviations.
        
    Parameters
    ----------
    stds : array_like, float
        Array of light curve standard deviations.
        
    Returns
    ----------
    stds_scaled : array_like, floats
        Scaled standard deviations
    """
    stds_scaled = StandardScaler().fit_transform(np.log10(stds).reshape(-1, 1))#This line computes the base-10 logarithm of each element in the stds array. Taking the logarithm can help in handling 
    #data that spans multiple orders of magnitude.
    #Reshapes the resulting array to ensure that it has a single column. The -1 in reshape(-1, 1) implies that NumPy should automatically determine the number of rows based on the size of the original 
    #array.
    return stds_scaled


def data_for_torch(data, stds, label, batch_size, device, log):
    """
    Prepare and batch data for torch.
        
    Parameters
    ----------
    data : array_like, floats
        Scaled light curve flux values.
    
    stds : array_like, float
        Scaled light curve standard deviations.
        
    label : array_like, floats
        Scaled stellar property array
        
    batch_size : int
        Training batch size.
        
    device : string
         The device on which the torch tensor will be allocated.
        
    log : _io.TextIOWrapper
        Log file.
        
    Returns
    ----------
    loader : torch.utils.data.dataloader.DataLoader
        Iterable to load batched data.
    """
    #The DataLoader efficiently loads data in mini-batches, which helps optimize memory usage and speeds up the training process.
    #It allows the model to process a small batch of data at a time, rather than loading the entire dataset into memory, which may not be feasible for large datasets.
    #During training, neural networks are typically trained using mini-batch gradient descent, 
    #where the model updates its weights based on the gradient computed on a mini-batch of training examples.
    X = torch.tensor(data.T, dtype=torch.float, device=device).unsqueeze(1)#This line creates a PyTorch tensor X from the transposed data (data.T). The data type of the tensor is set to torch.float, 
    #and it's allocated to the specified device. Additionally, unsqueeze(1) is called to add an extra dimension at index 1, 
    #effectively adding a channel dimension for each data point.
    STDS = torch.tensor(stds, dtype=torch.float, device=device)#This line creates a PyTorch tensor STDS from the scaled standard deviations (stds).
    Y = torch.tensor(label, dtype=torch.float, device=device)#This line creates a PyTorch tensor Y from the scaled labels (label).
    
    zipped = [[X[i], STDS[i], Y[i]] for i in range(len(X))]#This line zips the tensors X, STDS, and Y together into a list of lists, where each sublist contains the corresponding elements from X, STDS, 
    #and Y tensors for each data point.
    loader = DataLoader(zipped, batch_size=batch_size)#This line creates a DataLoader object from the zipped list of lists. It batches the data into mini-batches of size batch_size.

    n_batches = len(loader)
    batch_shape = [_[0].shape for _ in loader][0]
    print('%s batches with shape %s' % (n_batches,batch_shape), file=log)
    print('%s batches with shape %s' % (n_batches,batch_shape))
    return loader
