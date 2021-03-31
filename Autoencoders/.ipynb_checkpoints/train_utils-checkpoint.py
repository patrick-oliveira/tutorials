import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, utils, datasets, models

ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

def train_autoencoder(model, dataloaders, dataset_size, criterion, optimizer, scheduler = None, num_epochs = 10, view = True):
    print("{:7}  {:10}  {:6}\n".format("Epoch", "Stage", "Loss"))
    
    best_model_wgts = copy.deepcopy(model.state_dict())
    best_loss = np.inf
    train_loss = []
    validation_loss  = []
    stats = {"Train": train_loss,
             "Validation": validation_loss}
    
    since = time.time()
    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            cumulative_loss = 0.0
            for inputs, _ in dataloaders[phase]:
                if view:
                    inputs = inputs.reshape(inputs.shape[0], 1, -1)
                inputs = inputs.to(device)
#                 print(inputs.shape)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    
                    
                    
                    loss = criterion(outputs, inputs)
                    if len(loss) > 1:
                        reconstruction = loss[1]
                        loss = loss[0]
                        cumulative_loss += reconstruction.item()*inputs.size(0)
                    else:
                        cumulative_loss += loss.item()*inputs.size(0)
                    
                    penalty_loss = None
                    try:
                        penalty_loss = model.penalty(inputs)
                    except:
                        pass
                    
                    if penalty_loss != None:
                        loss += penalty_loss
                        
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                del(inputs); del(_)
                
            if phase == 'train' and scheduler != None:
                scheduler.step()
                
            epoch_loss = cumulative_loss / dataset_size[phase]
            train_loss.append(epoch_loss) if phase == 'train' else validation_loss.append(epoch_loss)
            print("{:7}  {:10}  {:<6.2f}".format("{}/{}".format(epoch + 1, num_epochs) if phase == "train" else " ",
                                                        "Training" if phase == "train" else "Validation",
                                                        epoch_loss))
            
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wgts = copy.deepcopy(model.state_dict())     
    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print("Best Validation Loss: {:.2f}\n".format(best_loss))
    
    stats["Best Validation Loss"] = best_loss
    model.load_state_dict(best_model_wgts)
    return model, stats