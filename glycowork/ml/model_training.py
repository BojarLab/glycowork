import copy
import time
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
try:
    import torch
    import torch.nn.functional as F
    # Choose the correct computing architecture
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
except ImportError:
    raise ImportError("<torch missing; did you do 'pip install glycowork[ml]'?>")
from sklearn.metrics import accuracy_score, matthews_corrcoef, mean_squared_error, \
    label_ranking_average_precision_score, ndcg_score, roc_auc_score, mean_absolute_error, r2_score
from glycowork.motif.annotate import annotate_dataset


class EarlyStopping:
    def __init__(self, patience: int = 7, # epochs to wait after last improvement
                 verbose: bool = False # whether to print messages
                ) -> None:
        "Early stops the training if validation loss doesn't improve after a given patience"
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = 0

    def __call__(self, val_loss: float, model: torch.nn.Module) -> None:
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

    def save_checkpoint(self, val_loss: float, model: torch.nn.Module) -> None:
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), 'drive/My Drive/checkpoint.pt')
        self.val_loss_min = val_loss


def sigmoid(x: float # input value
          ) -> float: # sigmoid transformed value
    "Apply sigmoid transformation to input"
    return 1 / (1 + math.exp(-x))


def disable_running_stats(model: torch.nn.Module # model to disable batch norm
                       ) -> None:
    "Disable batch normalization running statistics"

    def _disable(module):
        if isinstance(module, torch.nn.BatchNorm1d):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)


def enable_running_stats(model: torch.nn.Module # model to enable batch norm
                      ) -> None:
    "Enable batch normalization running statistics"

    def _enable(module):
        if isinstance(module, torch.nn.BatchNorm1d) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)


def train_model(model: torch.nn.Module, # graph neural network for analyzing glycans
               dataloaders: Dict[str, torch.utils.data.DataLoader], # dict with 'train' and 'val' loaders
               criterion: torch.nn.Module, # PyTorch loss function
               optimizer: torch.optim.Optimizer, # PyTorch optimizer, has to be SAM if mode != "regression"
               scheduler: torch.optim.lr_scheduler._LRScheduler, # PyTorch learning rate decay
               num_epochs: int = 25, # number of epochs for training
               patience: int = 50, # epochs without improvement until early stop
               mode: str = 'classification', # 'classification', 'multilabel', or 'regression'
               mode2: str = 'multi', # 'multi' or 'binary' classification
               return_metrics: bool = False, # whether to return metrics
              ) -> Union[torch.nn.Module, tuple[torch.nn.Module, dict[str, dict[str, list[float]]]]]: # best model from training and the training and validation metrics
    "trains a deep learning model on predicting glycan properties"

    since = time.time()
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    best_lead_metric = float("inf")

    if mode == 'classification':
        blank_metrics = {"loss": [], "acc": [], "mcc": [], "auroc": []}
    elif mode == 'multilabel':
        blank_metrics = {"loss": [], "acc": [], "mcc": [], "lrap": [], "ndcg": []}
    else:
        blank_metrics = {"loss": [], "mse": [], "mae": [], "r2": []}

    metrics = {"train": copy.deepcopy(blank_metrics), "val": copy.deepcopy(blank_metrics)}

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_metrics = copy.deepcopy(blank_metrics)
            running_metrics["weights"] = []

            for data in dataloaders[phase]:
                # Get all relevant node attributes; top LectinOracle-style models, bottom SweetNet-style models
                try:
                    x, y, edge_index, prot, batch = data.labels, data.y, data.edge_index, data.train_idx, data.batch
                    prot = prot.view(max(batch) + 1, -1).to(device)
                except:
                    x, y, edge_index, batch = data.labels, data.y, data.edge_index, data.batch
                x = x.to(device)
                if mode == 'multilabel':
                    y = y.view(max(batch) + 1, -1).to(device)
                else:
                    y = y.to(device)
                edge_index = edge_index.to(device)
                batch = batch.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # First forward pass
                    if mode + mode2 == 'classificationmulti' or mode + mode2 == 'multilabelmulti':
                        enable_running_stats(model)
                    try:
                        pred = model(prot, x, edge_index, batch)
                        loss = criterion(pred, y.view(-1, 1))
                    except:
                        pred = model(x, edge_index, batch)
                        loss = criterion(pred, y)

                    if phase == 'train':
                        loss.backward()
                        if mode + mode2 == 'classificationmulti' or mode + mode2 == 'multilabelmulti':
                            optimizer.first_step(zero_grad=True)
                            # Second forward pass
                            disable_running_stats(model)
                            try:
                                criterion(model(prot, x, edge_index, batch), y.view(-1, 1)).backward()
                            except:
                                criterion(model(x, edge_index, batch), y).backward()
                            optimizer.second_step(zero_grad=True)
                        else:
                            optimizer.step()

                # Collecting relevant metrics
                running_metrics["loss"].append(loss.item())
                running_metrics["weights"].append(batch.max().cpu() + 1)

                y_det = y.detach().cpu().numpy()
                pred_det = pred.cpu().detach().numpy()
                if mode == 'classification':
                    if mode2 == 'multi':
                        pred2 = np.argmax(pred_det, axis=1)
                    else:
                        pred2 = [np.round(sigmoid(x)) for x in pred_det]
                    running_metrics["acc"].append(accuracy_score(y_det.astype(int), pred2))
                    running_metrics["mcc"].append(matthews_corrcoef(y_det, pred2))
                    running_metrics["auroc"].append(roc_auc_score(y_det.astype(int), pred2))
                elif mode == 'multilabel':
                    running_metrics["acc"].append(accuracy_score(y_det.astype(int), pred_det))
                    running_metrics["mcc"].append(matthews_corrcoef(y_det, pred_det))
                    running_metrics["lrap"].append(label_ranking_average_precision_score(y_det.astype(int), pred_det))
                    running_metrics["ndcg"].append(ndcg_score(y_det.astype(int), pred_det))
                else:
                    running_metrics["mse"].append(mean_squared_error(y_det, pred_det))
                    running_metrics["mae"].append(mean_absolute_error(y_det, pred_det))
                    running_metrics["r2"].append(r2_score(y_det, pred_det))

            # Averaging metrics at end of epoch
            for key in running_metrics:
                if key == "weights":
                    continue
                metrics[phase][key].append(np.average(running_metrics[key], weights=running_metrics["weights"]))

            if mode == 'classification':
                print('{} Loss: {:.4f} Accuracy: {:.4f} MCC: {:.4f}'.format(phase, metrics[phase]["loss"][-1], metrics[phase]["acc"][-1], metrics[phase]["mcc"][-1]))
            elif mode == 'multilabel':
                print('{} Loss: {:.4f} LRAP: {:.4f} NDCG: {:.4f}'.format(phase, metrics[phase]["loss"][-1], metrics[phase]["acc"][-1], metrics[phase]["mcc"][-1]))
            else:
                print('{} Loss: {:.4f} MSE: {:.4f} MAE: {:.4f}'.format(phase, metrics[phase]["loss"][-1], metrics[phase]["mse"][-1], metrics[phase]["mae"][-1]))

            # Keep best model state_dict
            if phase == "val":
                if metrics[phase]["loss"][-1] <= best_loss:
                    best_loss = metrics[phase]["loss"][-1]
                    best_model_wts = copy.deepcopy(model.state_dict())

                    # Extract the lead metric (ACC, LRAP, or MSE) of the new best model
                    if mode == 'classification':
                        best_lead_metric = metrics[phase]["acc"][-1]
                    elif mode == 'multilabel':
                        best_lead_metric = metrics[phase]["lrap"][-1]
                    else:
                        best_lead_metric = metrics[phase]["mse"][-1]

                # Check Early Stopping & adjust learning rate if needed
                early_stopping(metrics[phase]["loss"][-1], model)
                try:
                    scheduler.step(metrics[phase]["loss"][-1])
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
        print('Best val loss: {:4f}, best Accuracy score: {:.4f}'.format(best_loss, best_lead_metric))
    elif mode == 'multilabel':
        print('Best val loss: {:4f}, best LRAP score: {:.4f}'.format(best_loss, best_lead_metric))
    else:
        print('Best val loss: {:4f}, best MSE score: {:.4f}'.format(best_loss, best_lead_metric))
    model.load_state_dict(best_model_wts)

    if return_metrics:
        return model, metrics

    # Plot loss & score over the course of training
    _, _ = plt.subplots(nrows=2, ncols=1)
    plt.subplot(2, 1, 1)
    plt.plot(range(epoch + 1), metrics["val"]["loss"])
    plt.title('Model Training')
    plt.ylabel('Validation Loss')
    plt.legend(['Validation Loss'], loc='best')

    plt.subplot(2, 1, 2)
    plt.xlabel('Number of Epochs')
    if mode == 'classification':
        plt.plot(range(epoch + 1), metrics["val"]["acc"])
        plt.ylabel('Validation Accuracy')
        plt.legend(['Validation Accuracy'], loc='best')
    elif mode == 'multilabel':
        plt.plot(range(epoch + 1), metrics["val"]["lrap"])
        plt.ylabel('Validation LRAP')
        plt.legend(['Validation LRAP'], loc='best')
    else:
        plt.plot(range(epoch + 1), metrics["val"]["mse"])
        plt.ylabel('Validation MSE')
        plt.legend(['Validation MSE'], loc='best')
    return model


class SAM(torch.optim.Optimizer):
    def __init__(self, params: List[torch.nn.Parameter], # model parameters
                 base_optimizer: type[torch.optim.Optimizer], # base PyTorch optimizer type
                 rho: float = 0.5, # size of neighborhood to explore
                 alpha: float = 0.0, # surrogate gap minimization coefficient
                 adaptive: bool = False, # whether to use adaptive SAM
                 **kwargs # additional optimizer arguments
                ) -> None:
        "Sharpness-Aware Minimization (SAM) optimizer adapted from https://github.com/davda54/sam"
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        assert alpha >= 0.0, f"Invalid alpha, should be non-negative: {alpha}"

        defaults = dict(rho = rho, alpha = alpha, adaptive = adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        self.minimize_surrogate_gap = any(group["alpha"] > 0.0 for group in self.param_groups)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False # whether to zero gradients after step
                 ) -> None:
        "Performs first optimization step to find adversarial weights"
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                if self.minimize_surrogate_gap:
                    self.state[p]["old_g"] = p.grad.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # Climb to the local maximum "w + e(w)"

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False # whether to zero gradients after step
                  ) -> None:
        "Performs second optimization step with regular weights"
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]  # Get back to "w" from "w + e(w)"

        if self.minimize_surrogate_gap:
            self._gradient_decompose()
        self.base_optimizer.step()  # Do the actual "sharpness-aware" update

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> None:
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # The closure should do a full forward-backward pass

        self.first_step(zero_grad = True)
        closure()
        self.second_step()

    def _gradient_decompose(self) -> None:
        coeff_nomin, coeff_denom = 0.0, 0.0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                coeff_nomin += (self.state[p]['old_g'] * p.grad).sum()
                coeff_denom += p.grad.pow(2).sum()

        coeff = coeff_nomin / (coeff_denom + 1e-12)

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                rejection = self.state[p]['old_g'] - coeff * p.grad
                p.grad.data.add_(rejection, alpha=-group["alpha"])

    def _grad_norm(self) -> torch.Tensor:
        shared_device = self.param_groups[0]["params"][0].device  # Put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p = 2
               )
        return norm

    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


class Poly1CrossEntropyLoss(torch.nn.Module):
    def __init__(self, num_classes: int, # number of classes
                 epsilon: float = 1.0, # weight of poly1 term
                 reduction: str = "mean", # reduction method for loss
                 weight: Optional[torch.Tensor] = None # manual class weights
                ) -> None:
        "Polynomial cross entropy loss for improved training stability"
        super(Poly1CrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.reduction = reduction
        self.weight = weight
        return

    def forward(self, logits: torch.Tensor, # predicted class probabilities [N, num_classes]
               labels: torch.Tensor # ground truth labels [N]
              ) -> torch.Tensor: # computed loss value
        "Compute poly cross-entropy loss"
        if len(labels.shape) == 2 and labels.shape[1] == self.num_classes:
            labels_onehot = labels.to(device = logits.device, dtype = logits.dtype)
            labels = torch.argmax(labels, dim = 1)
        else:
            labels_onehot = F.one_hot(labels, num_classes = self.num_classes).to(device = logits.device,
                                                                           dtype = logits.dtype)
        pt = torch.sum(labels_onehot * F.softmax(logits, dim = -1), dim = -1)
        CE = F.cross_entropy(input = logits,
                             target = labels,
                             reduction = 'none',
                             weight = self.weight,
                             label_smoothing = 0.1)
        poly1 = CE + self.epsilon * (1 - pt)
        if self.reduction == "mean":
            poly1 = poly1.mean()
        elif self.reduction == "sum":
            poly1 = poly1.sum()
        return poly1


def training_setup(model: torch.nn.Module, # graph neural network for analyzing glycans
                  lr: float, # learning rate
                  lr_patience: int = 4, # epochs before reducing learning rate
                  factor: float = 0.2, # factor to multiply lr on reduction
                  weight_decay: float = 0.0001, # regularization parameter
                  mode: str = 'multiclass', # type of prediction task
                  num_classes: int = 2, # number of classes for classification
                  gsam_alpha: float = 0. # if >0, uses GSAM instead of SAM optimizer
                 ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler, torch.nn.Module]: # optimizer, scheduler, criterion
    "prepares optimizer, learning rate scheduler, and loss criterion for model training"
    # Choose optimizer & learning rate scheduler
    if mode in {'multiclass', 'multilabel'}:
        optimizer_ft = SAM(model.parameters(), torch.optim.AdamW, alpha = gsam_alpha, lr = lr,
                           weight_decay = weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_ft.base_optimizer, patience = lr_patience,
                                                               factor = factor)
    else:
        optimizer_ft = torch.optim.AdamW(model.parameters(), lr = lr,
                                         weight_decay = weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_ft, patience = lr_patience,
                                                               factor = factor)
    # Choose loss function
    if mode == 'multiclass':
        if num_classes == 2:
            raise Exception("You have to set the number of classes via num_classes")
        criterion = Poly1CrossEntropyLoss(num_classes = num_classes).to(device)
    elif mode == 'multilabel':
        criterion = Poly1CrossEntropyLoss(num_classes = num_classes).to(device)
    elif mode == 'binary':
        criterion = Poly1CrossEntropyLoss(num_classes = 2).to(device)
    elif mode == 'regression':
        criterion = torch.nn.MSELoss().to(device)
    else:
        print("Invalid option. Please pass 'multiclass', 'multilabel', 'binary', or 'regression'.")
    return optimizer_ft, scheduler, criterion


def train_ml_model(X_train: Union[pd.DataFrame, List], # training data/glycans
                  X_test: Union[pd.DataFrame, List], # test data/glycans
                  y_train: List, # training labels
                  y_test: List, # test labels
                  mode: str = 'classification', # 'classification' or 'regression'
                  feature_calc: bool = False, # calculate motifs from glycans
                  return_features: bool = False, # return calculated features
                  feature_set: List[str] = ['known', 'exhaustive'], # feature set for annotations
                  additional_features_train: Optional[pd.DataFrame] = None, # additional training features
                  additional_features_test: Optional[pd.DataFrame] = None # additional test features
                 ) -> Union[xgb.XGBModel, Tuple[xgb.XGBModel, pd.DataFrame, pd.DataFrame]]: # trained model and optionally features
    "wrapper function to train standard machine learning models on glycans"
    # Choose model type
    if mode == 'classification':
        model = xgb.XGBClassifier(random_state = 42, n_estimators = 100,  max_depth = 3)
    elif mode == 'regression':
        model = xgb.XGBRegressor(random_state = 42, n_estimators = 100, objective = 'reg:squarederror')
    # Get features
    if feature_calc:
        print("\nCalculating Glycan Features...")
        X_train = annotate_dataset(X_train, feature_set = feature_set, condense = True)
        X_test = annotate_dataset(X_test, feature_set = feature_set, condense = True)
        # Get the difference between the columns
        missing_in_X_train = set(X_test.columns) - set(X_train.columns)
        missing_in_X_test = set(X_train.columns) - set(X_test.columns)
        # Fill in the missing columns
        for k in missing_in_X_train:
            X_train[k] = 0
        for k in missing_in_X_test:
            X_test[k] = 0
        X_train = X_train.apply(pd.to_numeric)
        X_test = X_test.apply(pd.to_numeric)
        if additional_features_train is not None:
            X_train = pd.concat([X_train, additional_features_train], axis = 1)
            X_test = pd.concat([X_test, additional_features_test], axis = 1)
    print("\nTraining model...")
    model.fit(X_train, y_train)
    # Keep track of column order & re-order test set accordingly
    cols_when_model_builds = model.get_booster().feature_names
    X_test = X_test[cols_when_model_builds]
    print("\nEvaluating model...")
    preds = model.predict(X_test)
    # Get metrics of trained model
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


def analyze_ml_model(model: xgb.XGBModel # trained ML model from train_ml_model
                   ) -> None:
    "plots relevant features for model prediction"
    # Get important features
    feat_imp = model.get_booster().get_score(importance_type = 'gain')
    feat_imp = pd.DataFrame(feat_imp, index = [0]).T
    feat_imp = feat_imp.sort_values(by = feat_imp.columns.values.tolist()[0], ascending = False)
    feat_imp = feat_imp[:10]
    # Plot important features
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


def get_mismatch(model: xgb.XGBModel, # trained ML model from train_ml_model
                X_test: pd.DataFrame, # motif dataframe for validation
                y_test: List, # test labels
                n: int = 10 # number of returned misclassifications
               ) -> List[Tuple[Any, float]]: # misclassifications and predicted probabilities
    "analyzes misclassifications of trained machine learning model"
    # Get predictions
    preds = model.predict(X_test)
    preds_proba = model.predict_proba(X_test)
    # Get false predictions
    idx = [k for k in range(len(preds)) if preds[k] != y_test[k]]
    preds = X_test.iloc[idx, :].index.values.tolist()
    preds_proba = [preds_proba[k].tolist()[1] for k in idx]
    # Return the mismatches
    mismatch = [tuple([i, j]) for i, j in zip(preds, preds_proba)]
    return mismatch[:min(len(mismatch), n)]
