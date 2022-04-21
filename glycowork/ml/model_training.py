import copy
import time
import math
import torch
import numpy as np
import pandas as pd
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, matthews_corrcoef, mean_squared_error, label_ranking_average_precision_score, ndcg_score
from glycowork.motif.annotate import annotate_dataset
from glycowork.glycan_data.loader import lib

#choose the correct computing architecture
device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"

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

def sigmoid(x):
       return 1 / (1 + math.exp(-x))

def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, torch.nn.BatchNorm1d):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)

def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, torch.nn.BatchNorm1d) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)

def train_model(model, dataloaders, criterion, optimizer,
                scheduler, num_epochs = 25, patience = 50,
                mode = 'classification', mode2 = 'multi'):
  """trains a deep learning model on predicting glycan properties\n
  | Arguments:
  | :-
  | model (PyTorch object): graph neural network (such as SweetNet) for analyzing glycans
  | dataloaders (PyTorch object): dictionary of dataloader objects with keys 'train' and 'val'
  | criterion (PyTorch object): PyTorch loss function
  | optimizer (PyTorch object): PyTorch optimizer
  | scheduler (PyTorch object): PyTorch learning rate decay
  | num_epochs (int): number of epochs for training; default:25
  | patience (int): number of epochs without improvement until early stop; default:50
  | mode (string): 'classification', 'multilabel', or 'regression'; default:classification
  | mode2 (string): further specifying classification into 'multi' or 'binary' classification;default:multi\n
  | Returns:
  | :-
  | Returns the best model seen during training
  """
  since = time.time()
  early_stopping = EarlyStopping(patience = patience, verbose = True)
  best_model_wts = copy.deepcopy(model.state_dict())
  best_loss = 100.0
  epoch_mcc = 0
  if mode != 'regression':
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
        #get all relevant node attributes; top LectinOracle-style models, bottom SweetNet-style models
        try:
            x, y, edge_index, prot, batch = data.labels, data.y, data.edge_index, data.train_idx, data.batch
            prot = prot.view(max(batch)+1, -1).to(device)
        except:
            x, y, edge_index, batch = data.labels, data.y, data.edge_index, data.batch
        x = x.to(device)
        y = y.view(max(batch)+1, -1).to(device)
        edge_index = edge_index.to(device)
        batch = batch.to(device)
        optimizer.zero_grad()

        with torch.set_grad_enabled(phase == 'train'):
          #first forward pass
          enable_running_stats(model)
          try:
              pred = model(prot, x, edge_index, batch)
              loss = criterion(pred, y.view(-1,1))
          except:
              pred = model(x, edge_index, batch)
              loss = criterion(pred, y)

          if phase == 'train':
            loss.backward()
            optimizer.first_step(zero_grad = True)
            #second forward pass
            disable_running_stats(model)
            try:
                criterion(model(prot, x, edge_index, batch), y).backward()
            except:
                criterion(model(x, edge_index, batch), y).backward()
            optimizer.second_step(zero_grad = True)

        #collecting relevant metrics            
        running_loss.append(loss.item())
        if mode == 'classification':
            if mode2 == 'multi':
                pred2 = np.argmax(pred.cpu().detach().numpy(), axis = 1)
            else:
                pred2 = [sigmoid(x) for x in pred.cpu().detach().numpy()]
                pred2 = [np.round(x) for x in pred2]
            running_acc.append(accuracy_score(
                                   y.cpu().detach().numpy().astype(int), pred2))
            running_mcc.append(matthews_corrcoef(y.detach().cpu().numpy(), pred2))
        elif mode == 'multilabel':
            running_acc.append(label_ranking_average_precision_score(y.cpu().detach().numpy().astype(int),
                                                                 pred.cpu().detach().numpy()))
            running_mcc.append(ndcg_score(y.cpu().detach().numpy().astype(int),
                                                                 pred.cpu().detach().numpy()))
        else:
            running_acc.append(mean_squared_error(y.cpu().detach().numpy(), pred.cpu().detach().numpy()))

      #averaging metrics at end of epoch  
      epoch_loss = np.mean(running_loss)
      epoch_acc = np.mean(running_acc)
      if mode != 'regression':
          epoch_mcc = np.mean(running_mcc)
      else:
          epoch_mcc = 0
      if mode == 'classification':
          print('{} Loss: {:.4f} Accuracy: {:.4f} MCC: {:.4f}'.format(
              phase, epoch_loss, epoch_acc, epoch_mcc))
      elif mode == 'multilabel':
          print('{} Loss: {:.4f} LRAP: {:.4f} NDCG: {:.4f}'.format(
              phase, epoch_loss, epoch_acc, epoch_mcc))
      else:
          print('{} Loss: {:.4f} MSE: {:.4f}'.format(
              phase, epoch_loss, epoch_acc))

      #keep best model state_dict
      if phase == 'val' and epoch_loss <= best_loss:
        best_loss = epoch_loss
        best_model_wts = copy.deepcopy(model.state_dict())
      if mode != 'regression':
          if phase == 'val' and epoch_acc > best_acc:
              best_acc = epoch_acc
      else:
          if phase == 'val' and epoch_acc < best_acc:
              best_acc = epoch_acc
      if phase == 'val':
        val_losses.append(epoch_loss)
        val_acc.append(epoch_acc)
        #check Early Stopping & adjust learning rate if needed
        early_stopping(epoch_loss, model)
        try:
            scheduler.step(epoch_loss)
        except:
            scheduler.step()
        
    if early_stopping.early_stop:
      print("Early stopping")
      break
    print()
    
  time_elapsed = time.time() - since
  print('Training complete in {:.0f}m {:.0f}s'.format(
      time_elapsed // 60, time_elapsed % 60))
  if mode == 'classification':
      print('Best val loss: {:4f}, best Accuracy score: {:.4f}'.format(best_loss, best_acc))
  elif mode == 'multilabel':
      print('Best val loss: {:4f}, best LRAP score: {:.4f}'.format(best_loss, best_acc))
  else:
      print('Best val loss: {:4f}, best MSE score: {:.4f}'.format(best_loss, best_acc))
  model.load_state_dict(best_model_wts)

  ## plot loss & score over the course of training 
  fig, ax = plt.subplots(nrows = 2, ncols = 1) 
  plt.subplot(2, 1, 1)
  plt.plot(range(epoch+1), val_losses)
  plt.title('Model Training')
  plt.ylabel('Validation Loss')
  plt.legend(['Validation Loss'],loc = 'best')

  plt.subplot(2, 1, 2)
  plt.plot(range(epoch+1), val_acc)
  plt.xlabel('Number of Epochs')
  if mode == 'classification':
      plt.ylabel('Validation Accuracy')
      plt.legend(['Validation Accuracy'], loc = 'best')
  elif mode == 'multilabel':
      plt.ylabel('Validation LRAP')
      plt.legend(['Validation LRAP'], loc = 'best')
  else:
      plt.ylabel('Validation MSE')
      plt.legend(['Validation MSE'], loc = 'best')
  return model

class SAM(torch.optim.Optimizer):
    #adapted from https://github.com/davda54/sam
    def __init__(self, params, base_optimizer, rho = 0.5, adaptive = False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho = rho, adaptive = adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad = False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure = None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad = True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p = 2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

def training_setup(model, lr, lr_patience = 4, factor = 0.2, weight_decay = 0.0001,
                   mode = 'multiclass'):
    """prepares optimizer, learning rate scheduler, and loss criterion for model training\n
    | Arguments:
    | :-
    | model (PyTorch object): graph neural network (such as SweetNet) for analyzing glycans
    | lr (float): learning rate
    | lr_patience (int): number of epochs without validation loss improvement before reducing the learning rate;default:4
    | factor (float): factor by which learning rate is multiplied upon reduction
    | weight_decay (float): regularization parameter for the optimizer; default:0.001
    | mode (string): 'multiclass': classification with multiple classes, 'multilabel': predicting several labels at the same time, 'binary':binary classification, 'regression': regression; default:'multiclass'\n
    | Returns:
    | :-
    | Returns optimizer, learning rate scheduler, and loss criterion objects
    """
    #choose optimizer
    optimizer_ft = SAM(model.parameters(), torch.optim.AdamW, lr = lr,
                       weight_decay = weight_decay)
    #choose learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_ft.base_optimizer, patience = lr_patience,
                                                           factor = factor, verbose = True)
    #choose loss function
    if mode == 'multiclass':
        criterion = torch.nn.CrossEntropyLoss().to(device)
    elif mode == 'multilabel':
        criterion = torch.nn.BCEWithLogitsLoss().to(device)
    elif mode == 'binary':
        criterion = torch.nn.BCEWithLogitsLoss().to(device)
    elif mode == 'regression':
        criterion = torch.nn.MSELoss().to(device)
    else:
        print("Invalid option. Please pass 'multiclass', 'multilabel', 'binary', or 'regression'.")
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
    | feature_set (list): which feature set to use for annotations, add more to list to expand; default:['known','exhaustive']; options are: 'known' (hand-crafted glycan features), 'graph' (structural graph features of glycans), and 'exhaustive' (all mono- and disaccharide features)
    | additional_features_train (dataframe): additional features (apart from glycans) to be used for training. Has to be of the same length as X_train; default:None
    | additional_features_test (dataframe): additional features (apart from glycans) to be used for evaluation. Has to be of the same length as X_test; default:None\n
    | Returns:
    | :-
    | Returns trained model                           
    """
    #choose model type
    if mode == 'classification':
        model = xgb.XGBClassifier(random_state = 42, n_estimators = 100,
                                  max_depth = 3)
    elif mode == 'regression':
        model = xgb.XGBRegressor(random_state = 42, n_estimators = 100,
                                 objective = 'reg:squarederror')
    #get features
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
    #keep track of column order & re-order test set accordingly
    cols_when_model_builds = model.get_booster().feature_names
    X_test = X_test[cols_when_model_builds]
    print("\nEvaluating model...")
    preds = model.predict(X_test)
    #get metrics of trained model
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
    #get important features
    feat_imp = model.get_booster().get_score(importance_type = 'gain')
    feat_imp = pd.DataFrame(feat_imp, index = [0]).T
    feat_imp = feat_imp.sort_values(by = feat_imp.columns.values.tolist()[0], ascending = False)
    feat_imp = feat_imp[:10]
    #plot important features
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
    #get predictions
    preds = model.predict(X_test)
    preds_proba = model.predict_proba(X_test)
    #get false predictions
    idx = [k for k in range(len(preds)) if preds[k] != y_test[k]]
    preds = X_test.iloc[idx, :].index.values.tolist()
    preds_proba = [preds_proba[k].tolist()[1] for k in idx]
    #return the mismatches
    mismatch = [tuple([i,j]) for i,j in zip(preds, preds_proba)]
    return mismatch[:min(len(mismatch),n)]
