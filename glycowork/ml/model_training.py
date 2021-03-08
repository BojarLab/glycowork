import copy
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, matthews_corrcoef, mean_squared_error

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience = 7, verbose = False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = 0

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        #torch.save(model.state_dict(), 'drive/My Drive/checkpoint.pt')
        self.val_loss_min = val_loss

def train_model(model, criterion, optimizer, scheduler, num_epochs = 25
                mode = 'classification'):
  since = time.time()
  best_model_wts = copy.deepcopy(model.state_dict())
  best_loss = 100.0
  epoch_mcc = 0
  if mode == 'classification':
      best_acc = 0.0
  else:
      best_acc = 100.0
  val_losses = []
  val_acc = []
  
  for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-'*10)
    
    for phase in ['train', 'val']:
      if phase == 'train':
        model.train()
      else:
        model.eval()
        
      running_loss = []
      running_acc = []
      running_mcc = []
      for data in dataloaders[phase]:
        x, y, edge_index, batch = data.x, data.y, data.edge_index, data.batch
        x = x.cuda()
        y = y.cuda()
        edge_index = edge_index.cuda()
        batch = batch.cuda()
        optimizer.zero_grad()

        with torch.set_grad_enabled(phase == 'train'):
          pred = model(x, edge_index, batch)
          loss = criterion(pred, y)

          if phase == 'train':
            loss.backward()
            optimizer.step()
            
        running_loss.append(loss.item())
        if mode == 'classification':
            pred2 = np.argmax(pred.cpu().detach().numpy(), axis = 1)
            running_acc.append(accuracy_score(
                                   y.cpu().detach().numpy().astype(int), pred2))
            running_mcc.append(matthews_corrcoef(y.detach().cpu().numpy(), pred2))
        else:
            running_acc.append(y.cpu().detach().numpy(), pred.cpu().detach().numpy())
        
      epoch_loss = np.mean(running_loss)
      epoch_acc = np.mean(running_acc)
      if mode == 'classification':
          epoch_mcc = np.mean(running_mcc)
      print('{} Loss: {:.4f} Accuracy: {:.4f} MCC: {:.4f}'.format(
          phase, epoch_loss, epoch_acc, epoch_mcc))
      
      if phase == 'val' and epoch_loss <= best_loss:
        best_loss = epoch_loss
        best_model_wts = copy.deepcopy(model.state_dict())
      if mode == 'classification':
          if phase == 'val' and epoch_acc > best_acc:
              best_acc = epoch_acc
      else:
          if phase == 'val' and epoch_acc < best_acc:
              best_acc = epoch_acc
      if phase == 'val':
        val_losses.append(epoch_loss)
        val_acc.append(epoch_acc)
        early_stopping(epoch_loss, model)

      #if epoch > swa_start:
      #  swa_model.update_parameters(model)
      #  swa_scheduler.step()
      #else:
      #  scheduler.step()
      scheduler.step()
        
    if early_stopping.early_stop:
      print("Early stopping")
      break
    print()
    
  time_elapsed = time.time() - since
  print('Training complete in {:.0f}m {:.0f}s'.format(
      time_elapsed // 60, time_elapsed % 60))
  print('Best val loss: {:4f}, best Accuracy score: {:.4f}'.format(best_loss, best_acc))
  model.load_state_dict(best_model_wts)

  ## plot loss & accuracy score over the course of training 
  fig, ax = plt.subplots(nrows = 2, ncols = 1) 
  plt.subplot(2, 1, 1)
  plt.plot(range(epoch+1), val_losses)
  plt.title('Training of SweetNet')
  plt.ylabel('Validation Loss')
  plt.legend(['Validation Loss'],loc = 'best')

  plt.subplot(2, 1, 2)
  plt.plot(range(epoch+1), val_acc)
  plt.ylabel('Validation Accuracy')
  plt.xlabel('Number of Epochs')
  plt.legend(['Validation Accuracy'], loc = 'best')
  return model
