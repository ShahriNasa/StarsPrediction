import os
import torch


def device_status(device):
    """Print device status"""
    print('Using device:', device)

    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:',
              round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ',
              round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')
    elif device.type == 'mps':
        print('Apple silicon (MPS)')
        print('Memory Usage:')
        print('Allocated:', round(torch.mps.current_allocated_memory() / 1024 ** 3, 1), 'GB')
        print("total GPU memory",round(torch.mps.driver_allocated_memory()/ 1024 ** 3, 1), 'GB')
        torch.mps.empty_cache()
        
    else:
        print('Unknown device type')


def create_log(dir, name):
    """
    Create log. If it already exists, delete it and overwrite.
        
    Parameters
    ----------
    dir : string
        Path to where to create log.
        
    name : string
        Name of the log.
        
    Returns
    ----------
    log : _io.TextIOWrapper
        Log file.
    """
    try:
        os.remove(dir+'%s.log' % name)
    except:
        pass
    log = open(dir+'%s.log' % name, 'a')
    return log

def create_log_path(dir, name):
   
    # Ensure dir ends with a slash if it's not empty
    if dir and not dir.endswith(os.sep):
        dir = dir + os.sep
    
    log_path = os.path.join(dir, f'{name}.log')
    
    try:
        os.remove(log_path)
    except FileNotFoundError:
        pass  # Ignore if the file does not exist
    
    return log_path