import copy
import time
import torch
import numpy as np
import pandas as pd
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, matthews_corrcoef, mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from glycowork.motif.annotate import annotate_dataset
from glycowork.glycan_data.loader import lib

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

def train_model(model, dataloaders, criterion, optimizer,
                scheduler, num_epochs = 25, patience = 50,
                mode = 'classification'):
  """trains a deep learning model on predicting glycan properties\n
  | Arguments:
  | :-
  | model (PyTorch object): graph neural network (such as SweetNet) for analyzing glycans
  | dataloaders (PyTorch object): dictionary of dataloader objects with keys 'train' and 'val'
  | criterion (PyTorch object): PyTorch loss function
  | optimizer (PyTorch object): PyTorch optimizer
  | scheduler (PyTorch object): PyTorch learning rate decay
  | num_epochs (int): number of epochs for training; default: 25
  | patience (int): number of epochs without improvement until early stop; default: 50
  | mode (string): 'classification' or 'regression'; default is binary classification\n
  | Returns:
  | :-
  | Returns the best model seen during training
  """
  since = time.time()
  early_stopping = EarlyStopping(patience = patience, verbose = True)
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
        try:
            x, y, edge_index, prot, batch = data.x, data.y, data.edge_index, data.train_idx, data.batch
            prot = prot.view(max(batch)+1, -1).cuda()
        except:
            x, y, edge_index, batch = data.x, data.y, data.edge_index, data.batch
        x = x.cuda()
        y = y.cuda()
        edge_index = edge_index.cuda()
        batch = batch.cuda()
        optimizer.zero_grad()

        with torch.set_grad_enabled(phase == 'train'):
          try:
              pred = model(prot, x, edge_index, batch)
              loss = criterion(pred, y.view(-1,1))
          except:
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

def training_setup(model, epochs, lr, lr_decay_length = 0.5, weight_decay = 0.001,
                   mode = 'multiclass'):
    """prepares optimizer, learning rate scheduler, and loss criterion for model training\n
    | Arguments:
    | :-
    | model (PyTorch object): graph neural network (such as SweetNet) for analyzing glycans
    | epochs (int): number of epochs for training the model
    | lr (float): learning rate
    | lr_decay_length (float): proportion of epochs over which to decay the learning rate;default:0.5
    | weight_decay (float): regularization parameter for the optimizer; default:0.001
    | mode (string): 'multiclass': classification with multiple classes, 'binary':binary classification, 'regression': regression; default:'multiclass'\n
    | Returns:
    | :-
    | Returns optimizer, learning rate scheduler, and loss criterion objects
    """
    lr_decay = np.round(epochs * lr_decay_length)
    optimizer_ft = torch.optim.Adam(model.parameters(), lr = lr,
                                    weight_decay = weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, lr_decay)
    if mode == 'multiclass':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    elif mode == 'binary':
        criterion = torch.nn.BCEWithLogitsLoss().cuda()
    elif mode == 'regression':
        criterion = torch.nn.MSELoss().cuda()
    else:
        print("Invalid option. Please pass 'multiclass', 'binary', or 'regression'.")
    return optimizer_ft, scheduler, criterion

def train_ml_model(X_train, X_test, y_train, y_test, mode = 'classification',
                   feature_calc = False, libr = None, return_features = False,
                   feature_set = ['known','exhaustive'], additional_features_train = None,
                   additional_features_test = None):
    """wrapper function to train standard machine learning models on glycans\n
    | Arguments:
    | :-
    | X_train, X_test (list or dataframe): either lists of glycans (needs feature_calc = True) or motif dataframes such as from annotate_dataset
    | y_train, y_test (list): lists of labels
    | mode (string): 'classification' or 'regression'; default:'classification'
    | feature_calc (bool): set to True for calculating motifs from glycans; default:False
    | libr (list): sorted list of unique glycoletters observed in the glycans of our data; default:lib
    | return_features (bool): whether to return calculated features; default:False
    | feature_set (list): which feature set to use for annotations, add more to list to expand; default:['known','exhaustive']; options are: 'known' (hand-crafted glycan features), 'graph' (structural graph features of glycans) and 'exhaustive' (all mono- and disaccharide features)\n
    | additional_features_train (dataframe): additional features (apart from glycans) to be used for training. Has to be of the same length as X_train; default:None
    | additional_features_test (dataframe): additional features (apart from glycans) to be used for evaluation. Has to be of the same length as X_test; default:None
    | Returns:
    | :-
    | Returns trained model                           
    """
    if mode == 'classification':
        model = xgb.XGBClassifier(random_state = 42, n_estimators = 100,
                                  max_depth = 3)
    elif mode == 'regression':
        model = xgb.XGBRegressor(random_state = 42, n_estimators = 100,
                                 objective = 'reg:squarederror')
    if feature_calc:
        print("\nCalculating Glycan Features...")
        if libr is None:
            libr = lib
        X_train = annotate_dataset(X_train, libr = libr, feature_set = feature_set,
                                   condense = True)
        X_test = annotate_dataset(X_test, libr = libr, feature_set = feature_set,
                                   condense = True)
        for k in X_test.columns.values.tolist():
            if k not in X_train.columns.values.tolist():
                X_train[k] = [0]*len(X_train)
        for k in X_train.columns.values.tolist():
            if k not in X_test.columns.values.tolist():
                X_test[k] = [0]*len(X_test)
        X_train = X_train.apply(pd.to_numeric)
        X_test = X_test.apply(pd.to_numeric)
        if additional_features_train is not None:
            X_train = pd.concat([X_train, additional_features_train], axis = 1)
            X_test = pd.concat([X_test, additional_features_test], axis = 1)
    print("\nTraining model...")
    model.fit(X_train, y_train)
    cols_when_model_builds = model.get_booster().feature_names
    X_test = X_test[cols_when_model_builds]
    print("\nEvaluating model...")
    preds = model.predict(X_test)
    if mode == 'classification':
        out = accuracy_score(y_test, preds)
        print("Accuracy of trained model on separate validation set: " + str(out))
    elif mode == 'regression':
        out = mean_squared_error(y_test, preds)
        print("Mean squared error of trained model on separate validation set: " + str(out))
    if return_features:
        return model, X_train, X_test
    else:
        return model

def analyze_ml_model(model):
    """plots relevant features for model prediction\n
    | Arguments:
    | :-
    | model (model object): trained machine learning model from train_ml_model
    """
    feat_imp = model.get_booster().get_score(importance_type = 'gain')
    feat_imp = pd.DataFrame(feat_imp, index = [0]).T
    feat_imp = feat_imp.sort_values(by = feat_imp.columns.values.tolist()[0], ascending = False)
    feat_imp = feat_imp[:10]
    sns.barplot(x = feat_imp.index.values.tolist(),
                y = feat_imp[feat_imp.columns.values.tolist()[0]],
                color = 'cornflowerblue')
    sns.despine(left = True, bottom = True)
    plt.xticks(rotation = 90)
    plt.xlabel('Important Variables')
    plt.ylabel('Relative Importance (Gain)')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()

def get_mismatch(model, X_test, y_test, n = 10):
    """analyzes misclassifications of trained machine learning model\n
    | Arguments:
    | :-
    | model (model object): trained machine learning model from train_ml_model
    | X_test (dataframe): motif dataframe used for validating model
    | y_test (list): list of labels
    | n (int): number of returned misclassifications; default:10\n
    | Returns:
    | :-
    | Returns tuples of misclassifications and their predicted probability
    """
    preds = model.predict(X_test)
    preds_proba = model.predict_proba(X_test)
    idx = [k for k in range(len(preds)) if preds[k] != y_test[k]]
    preds = X_test.iloc[idx, :].index.values.tolist()
    preds_proba = [preds_proba[k].tolist()[1] for k in idx]
    mismatch = [tuple([i,j]) for i,j in zip(preds, preds_proba)]
    return mismatch[:min(len(mismatch),n)]
