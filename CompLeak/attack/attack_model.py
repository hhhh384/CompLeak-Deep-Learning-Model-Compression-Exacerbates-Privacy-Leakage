import torch.nn.functional as F
from torch import nn
import logging
import joblib
import random
import numpy as np
from sklearn.metrics import roc_curve, auc
from scipy.interpolate import interp1d
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from tqdm import tqdm


class DT:
    def __init__(self):
        self.model = DecisionTreeClassifier(max_leaf_nodes=10, random_state=0)

    def train_model(self, train_x, train_y, save_name=None):
        self.model.fit(train_x, train_y)
        if save_name is not None:
            joblib.dump(self.model, save_name, compress=9)

    def load_model(self, save_name):
        self.model = joblib.load(save_name)
        return self.model

    def predict_proba(self, test_x):
        return self.model.predict_proba(test_x)

    def test_model_acc(self, test_x, test_y):
        pred_y = self.model.predict(test_x)
        pred_proba = self.model.predict_proba(test_x)
        return accuracy_score(test_y, pred_y),pred_y,pred_proba

    def test_model_auc(self, test_x, test_y):
        pred_y = self.model.predict_proba(test_x)

        fpr, tpr, thresholds = roc_curve(test_y, pred_y[:, 1])  
        
        fpr_target = 0.001
        interp_func = interp1d(fpr, tpr)
        tpr_at_low_fpr = interp_func(fpr_target)
        auc1 = auc(fpr, tpr)
        return auc1, tpr_at_low_fpr




class RF:
    def __init__(self, min_samples_leaf=30):
        self.model = RandomForestClassifier(random_state=0, n_estimators=500, min_samples_leaf=min_samples_leaf)

    def train_model(self, train_x, train_y, save_name=None):
        self.model.fit(train_x, train_y)
        if save_name is not None:
            joblib.dump(self.model, save_name, compress=9)

    def load_model(self, save_name):
        self.model = joblib.load(save_name)
        return self.model

    def predict_proba(self, test_x):
        return self.model.predict_proba(test_x)

    def test_model_acc(self, test_x, test_y):
        pred_y = self.model.predict(test_x)
        pred_proba = self.model.predict_proba(test_x)
        return accuracy_score(test_y, pred_y),pred_y,pred_proba 

    def test_model_auc(self, test_x, test_y):
        pred_y = self.model.predict_proba(test_x)
        fpr, tpr, thresholds = roc_curve(test_y, pred_y[:, 1])  
        
        fpr_target = 0.001
        interp_func = interp1d(fpr, tpr)
        tpr_at_low_fpr = interp_func(fpr_target)
        auc1 = auc(fpr, tpr)
        return auc1, tpr_at_low_fpr*100


class MLP:

    def __init__(self, random_seed=42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
        random.seed(random_seed)
        self.model = MLPClassifier(early_stopping=True, learning_rate_init=0.01, random_state=random_seed)


    def scaler_data(self, data):
        scaler = StandardScaler()
        scaler.fit(data)
        data = scaler.transform(data)
        return data

    def train_model(self, train_x, train_y, save_name=None):
        self.model.fit(train_x, train_y)
        if save_name is not None:
            joblib.dump(self.model, save_name, compress=9)

    def load_model(self, save_name):
        self.model = joblib.load(save_name)
        return self.model

    def predict_proba(self, test_x):
        return self.model.predict_proba(test_x)

    def test_model_acc(self, test_x, test_y):
        pred_y = self.model.predict(test_x)
        pred_proba = self.model.predict_proba(test_x)
        return accuracy_score(test_y, pred_y),pred_y,pred_proba

    def test_model_auc(self, test_x, test_y):
        pred_y = self.model.predict_proba(test_x)
        fpr, tpr, thresholds = roc_curve(test_y, pred_y[:, 1])  
        
        fpr_target = 0.001
        interp_func = interp1d(fpr, tpr)
        tpr_at_low_fpr = interp_func(fpr_target)
        auc1 = auc(fpr, tpr)
        return auc1, tpr_at_low_fpr


class LR:
    def __init__(self):
        self.model = LogisticRegression(random_state=0, solver='lbfgs', max_iter=400)

    def train_model(self, train_x, train_y, save_name=None):
        self.scaler = preprocessing.StandardScaler().fit(train_x)
        self.model.fit(self.scaler.transform(train_x), train_y)
        if save_name is not None:
            joblib.dump(self.model, save_name, compress=9)

    def load_model(self, save_name):
        self.model = joblib.load(save_name)
        return self.model

    def predict_proba(self, test_x):
        self.scaler = preprocessing.StandardScaler().fit(test_x)
        return self.model.predict_proba(self.scaler.transform(test_x))

    def test_model_acc(self, test_x, test_y):
        # self.load_model(model)
        pred_proba = self.model.predict_proba(test_x)
        pred_y = self.model.predict(self.scaler.transform(test_x))

        return accuracy_score(test_y, pred_y),pred_y,pred_proba

    def test_model_auc(self, test_x, test_y):
        pred_y = self.model.predict_proba(self.scaler.transform(test_x))
        fpr, tpr, thresholds = roc_curve(test_y, pred_y[:, 1])
        
        fpr_target = 0.001
        interp_func = interp1d(fpr, tpr)
        tpr_at_low_fpr = interp_func(fpr_target)
        auc1 = auc(fpr, tpr)
        return auc1, tpr_at_low_fpr*100


