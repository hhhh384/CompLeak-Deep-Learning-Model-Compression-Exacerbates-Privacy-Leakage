import torch
import torch.nn.functional as F
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tqdm import tqdm
import pickle
from base_model import BaseModel
import pandas as pd
import numpy as np
from attack.attack_utils import ConstructFeature, Attack, ThresholdAttacker, get_ratio, compute_quantiles, calculate_tpr_at_fpr
from data_utils import get_dataset
from torch.utils.data import TensorDataset, DataLoader, Dataset, ConcatDataset
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score

class MiaAttack:
    def __init__(self, victim_model, victim_compress_model, victim_train_loader, victim_test_loader,
                 shadow_model_list, shadow_compress_model_list, shadow_train_loader_list, shadow_test_loader_list, 
                 original = False, compress = True, num_cls=10, batch_size=128, save_folder_compress= "", device="cuda", attack_device="cuda", lr=0.001, optimizer="sgd", weight_decay=5e-4):
        self.victim_model = victim_model
        self.victim_compress_model = victim_compress_model
        self.victim_train_loader = victim_train_loader
        self.victim_test_loader = victim_test_loader
        self.shadow_model_list = shadow_model_list
        self.shadow_compress_model_list = shadow_compress_model_list
        self.shadow_train_loader_list = shadow_train_loader_list
        self.shadow_test_loader_list = shadow_test_loader_list
        self.num_cls = num_cls
        self.device = device
        self.attack_device = attack_device
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.save_folder_compress = save_folder_compress
        self.original= original
        self.compress = compress
        self._prepare()

    def _prepare(self):
        if self.compress:
            attack_compress_in_predicts_list, attack_compress_out_predicts_list = [], []
            attack_in_targets_list = []
            attack_out_targets_list= []
            attack_compress_in_sens_list, attack_compress_out_sens_list = [], []
            for shadow_compress_model, shadow_train_loader, shadow_test_loader in zip(
                self.shadow_compress_model_list, self.shadow_train_loader_list, self.shadow_test_loader_list):
                attack_compress_in_predicts, attack_in_targets, attack_compress_in_sens = shadow_compress_model.predict_target_sensitivity(shadow_train_loader)
                attack_compress_out_predicts, attack_out_targets, attack_compress_out_sens = shadow_compress_model.predict_target_sensitivity(shadow_test_loader)
                attack_compress_in_predicts_list.append(attack_compress_in_predicts)
                attack_in_targets_list.append(attack_in_targets)
                attack_out_targets_list.append(attack_out_targets)         
                attack_compress_out_predicts_list.append(attack_compress_out_predicts)
                attack_compress_in_sens_list.append(attack_compress_in_sens)        
                attack_compress_out_sens_list.append(attack_compress_out_sens)

            self.attack_compress_in_predicts = torch.cat(attack_compress_in_predicts_list, dim=0)
            self.attack_compress_out_predicts = torch.cat(attack_compress_out_predicts_list, dim=0)
            self.attack_compress_in_sens = torch.cat(attack_compress_in_sens_list, dim=0)
            self.attack_compress_out_sens = torch.cat(attack_compress_out_sens_list, dim=0)
            self.attack_in_targets = torch.cat(attack_in_targets_list, dim=0)
            self.attack_out_targets = torch.cat(attack_out_targets_list, dim=0)
            self.victim_compress_in_predicts, self.victim_in_targets,  self.victim_compress_in_sens = self.victim_compress_model.predict_target_sensitivity(self.victim_train_loader)
            self.victim_compress_out_predicts, self.victim_out_targets, self.victim_compress_out_sens = self.victim_compress_model.predict_target_sensitivity(self.victim_test_loader)


        if self.original:
            attack_original_in_predicts_list, attack_original_out_predicts_list = [], []
            attack_original_in_sens_list, attack_original_out_sens_list = [], []
            attack_in_targets_list = []
            attack_out_targets_list= []
            for shadow_model, shadow_train_loader, shadow_test_loader in zip(
                self.shadow_model_list, self.shadow_train_loader_list, self.shadow_test_loader_list):

                attack_original_in_predicts, attack_in_targets, attack_original_in_sens = shadow_model.predict_target_sensitivity(shadow_train_loader)
                attack_original_out_predicts, attack_out_targets, attack_original_out_sens = shadow_model.predict_target_sensitivity(shadow_test_loader)
                

                attack_original_in_predicts_list.append(attack_original_in_predicts)
                attack_in_targets_list.append(attack_in_targets)
                attack_out_targets_list.append(attack_out_targets)
                attack_original_out_predicts_list.append(attack_original_out_predicts)            
                attack_original_in_sens_list.append(attack_original_in_sens)
                attack_original_out_sens_list.append(attack_original_out_sens)            


            self.attack_original_in_predicts = torch.cat(attack_original_in_predicts_list, dim=0)
            self.attack_original_out_predicts = torch.cat(attack_original_out_predicts_list, dim=0)


            self.attack_original_in_sens = torch.cat(attack_original_in_sens_list, dim=0)
            self.attack_original_out_sens = torch.cat(attack_original_out_sens_list, dim=0)
            self.attack_in_targets = torch.cat(attack_in_targets_list, dim=0)
            self.attack_out_targets = torch.cat(attack_out_targets_list, dim=0)

            self.victim_original_in_predicts, self.victim_in_targets, self.victim_original_in_sens = self.victim_model.predict_target_sensitivity(self.victim_train_loader)
            self.victim_original_out_predicts, self.victim_out_targets, self.victim_original_out_sens = self.victim_model.predict_target_sensitivity(self.victim_test_loader)
    def construct_feature(self, posterior_df, attack_type):
        feature = ConstructFeature(posterior_df)
        if attack_type == 'SR':
            for method in ['CompLeakSR1','CompLeakSR2']:
                feature.obtain_feature(method, posterior_df)
        elif attack_type == 'NR':
            if self.original:
                for method in  ['CompLeakNR_train_based_method_1(target_original_model)', 'CompLeakNR_train_based_method_2(target_original_model)']:
                    feature.obtain_feature(method, posterior_df)
            if self.compress:
                for method in  ['CompLeakNR_train_based_method_1(target_compressed_model)', 'CompLeakNR_train_based_method_2(target_compressed_model)']:
                    feature.obtain_feature(method, posterior_df)

    def _save_posterior(self, posterior_df, save_path):
        pickle.dump(posterior_df, open(save_path, 'wb'))

    def _load_posterior(self, save_path):
        return pickle.load(open(save_path, 'rb'))



    def feature_prepare_SR(self):
        self.shadow_posterior_df = pd.DataFrame(columns=["original", "compress", "targets", "label"])
        self.attack_in_targets = F.one_hot(self.attack_in_targets, num_classes=self.num_cls).float()
        for index in range(self.attack_original_in_predicts.shape[0]):
            self.shadow_posterior_df.loc[len(self.shadow_posterior_df)] = [self.attack_original_in_predicts[index].cpu().reshape([1, -1]), self.attack_compress_in_predicts[index].cpu().reshape([1, -1]), self.attack_in_targets[index].cpu().reshape([1,-1]),1]
        

        self.attack_out_targets = F.one_hot(self.attack_out_targets, num_classes=self.num_cls).float()
        for index in range(self.attack_original_out_predicts.shape[0]):
            self.shadow_posterior_df.loc[len(self.shadow_posterior_df)] = [self.attack_original_out_predicts[index].cpu().reshape([1, -1]), self.attack_compress_out_predicts[index].cpu().reshape([1, -1]),self.attack_out_targets[index].cpu().reshape([1,-1]), 0]
        
        

        self.victim_posterior_df = pd.DataFrame(columns=["original", "compress", "targets", "label"])
        self.victim_in_targets = F.one_hot(self.victim_in_targets, num_classes=self.num_cls).float()
        for index in range(self.victim_original_in_predicts.shape[0]):
            self.victim_posterior_df.loc[len(self.victim_posterior_df)] = [self.victim_original_in_predicts[index].cpu().reshape([1, -1]), self.victim_compress_in_predicts[index].cpu().reshape([1, -1]), self.victim_in_targets[index].cpu().reshape([1,-1]),1]
        self.victim_out_targets = F.one_hot(self.victim_out_targets, num_classes=self.num_cls).float()
        for index in range(self.victim_original_out_predicts.shape[0]):
            self.victim_posterior_df.loc[len(self.victim_posterior_df)] = [self.victim_original_out_predicts[index].cpu().reshape([1, -1]), self.victim_compress_out_predicts[index].cpu().reshape([1, -1]),self.victim_out_targets[index].cpu().reshape([1,-1]), 0]
    
    

    def feature_prepare_NR(self):
        if self.original:
            self.shadow_posterior_df = pd.DataFrame(columns=["original", "targets", "label"])
            self.attack_in_targets = F.one_hot(self.attack_in_targets, num_classes=self.num_cls).float()
            for index in range(self.attack_original_in_predicts.shape[0]):
                self.shadow_posterior_df.loc[len(self.shadow_posterior_df)] = [self.attack_original_in_predicts[index].cpu().reshape([1, -1]), self.attack_in_targets[index].cpu().reshape([1,-1]),1]
            

            self.attack_out_targets = F.one_hot(self.attack_out_targets, num_classes=self.num_cls).float()
            for index in range(self.attack_original_out_predicts.shape[0]):
                self.shadow_posterior_df.loc[len(self.shadow_posterior_df)] = [self.attack_original_out_predicts[index].cpu().reshape([1, -1]), self.attack_out_targets[index].cpu().reshape([1,-1]), 0]
            
            

            self.victim_posterior_df = pd.DataFrame(columns=["original", "targets", "label"])
            self.victim_in_targets = F.one_hot(self.victim_in_targets, num_classes=self.num_cls).float()
            for index in range(self.victim_original_in_predicts.shape[0]):
                self.victim_posterior_df.loc[len(self.victim_posterior_df)] = [self.victim_original_in_predicts[index].cpu().reshape([1, -1]), self.victim_in_targets[index].cpu().reshape([1,-1]),1]
            self.victim_out_targets = F.one_hot(self.victim_out_targets, num_classes=self.num_cls).float()
            for index in range(self.victim_original_out_predicts.shape[0]):
                self.victim_posterior_df.loc[len(self.victim_posterior_df)] = [self.victim_original_out_predicts[index].cpu().reshape([1, -1]), self.victim_out_targets[index].cpu().reshape([1,-1]), 0]
        
        if self.compress:
            self.shadow_posterior_df = pd.DataFrame(columns=["compress", "targets", "label"])
            self.attack_in_targets = F.one_hot(self.attack_in_targets, num_classes=self.num_cls).float()
            for index in range(self.attack_compress_in_predicts.shape[0]):
                self.shadow_posterior_df.loc[len(self.shadow_posterior_df)] = [self.attack_compress_in_predicts[index].cpu().reshape([1, -1]), self.attack_in_targets[index].cpu().reshape([1,-1]),1]
            

            self.attack_out_targets = F.one_hot(self.attack_out_targets, num_classes=self.num_cls).float()
            for index in range(self.attack_compress_out_predicts.shape[0]):
                self.shadow_posterior_df.loc[len(self.shadow_posterior_df)] = [self.attack_compress_out_predicts[index].cpu().reshape([1, -1]),self.attack_out_targets[index].cpu().reshape([1,-1]), 0]
            

            self.victim_posterior_df = pd.DataFrame(columns=["compress", "targets", "label"])
            self.victim_in_targets = F.one_hot(self.victim_in_targets, num_classes=self.num_cls).float()
            for index in range(self.victim_compress_in_predicts.shape[0]):
                self.victim_posterior_df.loc[len(self.victim_posterior_df)] = [self.victim_compress_in_predicts[index].cpu().reshape([1, -1]), self.victim_in_targets[index].cpu().reshape([1,-1]),1]
            self.victim_out_targets = F.one_hot(self.victim_out_targets, num_classes=self.num_cls).float()
            for index in range(self.victim_compress_out_predicts.shape[0]):
                self.victim_posterior_df.loc[len(self.victim_posterior_df)] = [ self.victim_compress_out_predicts[index].cpu().reshape([1, -1]),self.victim_out_targets[index].cpu().reshape([1,-1]), 0]
        
    def CompLeakSR(self):
        self.feature_prepare_SR()
        self.construct_feature(self.shadow_posterior_df, 'SR')
        self.construct_feature(self.victim_posterior_df, 'SR')
        results_df = pd.DataFrame()
        for method in ['CompLeakSR1','CompLeakSR2']:
            print("{}:".format(method))
            for attack_model_name in ['LR', 'RF']:
                attack = Attack(attack_model_name, self.shadow_posterior_df, self.victim_posterior_df)
                path = f"{self.save_folder_compress}/{attack_model_name}"
                train_acc, train_auc, train_prob,_ = attack.train_attack_model(method, path)
                test_acc, test_auc, pred, test_prob, test_tpr_at_low_fpr = attack.test_attack_model(method)
                results_df = pd.concat([results_df, pd.DataFrame({
                        "attack_model_name": [attack_model_name],
                        "method": [method],
                        "acc": [test_acc],
                        "auc": [test_auc],
                        "tpr_at_low_fpr": [test_tpr_at_low_fpr],
                        "train_prob": [[float(x) for x in train_prob.flatten()]],
                        "test_prob": [[float(x) for x in test_prob.flatten()]],
                        "predictions": [[float(x) for x in pred.flatten()]]
                    })])
                results_df.to_csv(f"{self.save_folder_compress}/{attack_model_name}_CompLeakSR.csv", index=False)
        
        
    def CompLeakNR_train_based(self):
        self.feature_prepare_NR()
        self.construct_feature(self.shadow_posterior_df, 'NR')
        self.construct_feature(self.victim_posterior_df, 'NR')
        results_df = pd.DataFrame()
        if self.original:
            for method in ['CompLeakNR_train_based_method_1(target_original_model)', 'CompLeakNR_train_based_method_2(target_original_model)']:
                print("{}:".format(method))
                for attack_model_name in ['LR', 'RF']:
                    attack = Attack(attack_model_name, self.shadow_posterior_df, self.victim_posterior_df)
                    path = f"{self.save_folder_compress}/{attack_model_name}_"
                    train_acc, train_auc, train_prob, _ = attack.train_attack_model(method, path)
                    test_acc, test_auc, pred, test_prob, test_tpr_at_low_fpr = attack.test_attack_model(method)
        

        if self.compress:
            for method in ['CompLeakNR_train_based_method_1(target_compressed_model)', 'CompLeakNR_train_based_method_2(target_compressed_model)']:
                print("{}:".format(method))
                for attack_model_name in ['LR', 'RF']:
                    attack = Attack(attack_model_name, self.shadow_posterior_df, self.victim_posterior_df)
                    path = f"{self.save_folder_compress}/{attack_model_name}_"
                    train_acc, train_auc, train_prob, _ = attack.train_attack_model(method, path)
                    test_acc, test_auc, pred, test_prob, test_tpr_at_low_fpr = attack.test_attack_model(method)

    def CompLeakNR_metric_based(self):
        if self.original:
            original_attacker = ThresholdAttacker((self.attack_original_in_predicts.numpy(), self.attack_in_targets.cpu().numpy()),
                                    (self.attack_original_out_predicts.numpy(), self.attack_out_targets.cpu().numpy()),
                                    (self.victim_original_in_predicts.numpy() , self.victim_in_targets.cpu().numpy()),
                                    (self.victim_original_out_predicts.numpy(), self.victim_out_targets.cpu().numpy()),
                                    self.num_cls)
            _, _, _, entropy, ent_auc, ent_tpr_at_low_fpr, modentr, mod_auc, mod_tpr_at_low_fpr = original_attacker._mem_inf_benchmarks()
            print(f"CompLeakNR_metric_based_method_1(target_orginal_model):")
            print("attack_acc: %.3f%% | attack_auc: %.3f%% |attack_tpr_at_0.001_fpr: %.3f%%" % (entropy*100, ent_auc*100, ent_tpr_at_low_fpr*100))

            print(f"CompLeakNR_metric_based_method_2(target_orginal_model):")
            print("attack_acc: %.3f%% | attack_auc: %.3f%% |attack_tpr_at_0.001_fpr: %.3f%%" % (modentr*100, mod_auc*100, mod_tpr_at_low_fpr*100))

        if self.compress:
            compress_attacker = ThresholdAttacker((self.attack_compress_in_predicts.numpy(), self.attack_in_targets.cpu().numpy()),
                                    (self.attack_compress_out_predicts.numpy(), self.attack_out_targets.cpu().numpy()),
                                    (self.victim_compress_in_predicts.numpy() , self.victim_in_targets.cpu().numpy()),
                                    (self.victim_compress_out_predicts.numpy(), self.victim_out_targets.cpu().numpy()),
                                    self.num_cls)
            _, _, _, c_entropy, c_ent_auc, c_ent_tpr_at_low_fpr, c_modentr, c_mod_auc, c_mod_tpr_at_low_fpr = compress_attacker._mem_inf_benchmarks()
            print(f"CompLeakNR_metric_based_method_1(target_compressed_model):")
            print("attack_acc: %.3f%% | attack_auc: %.3f%% |attack_tpr_at_0.001_fpr: %.3f%%" % (c_entropy*100, c_ent_auc*100, c_ent_tpr_at_low_fpr*100))

            print(f"CompLeakNR_metric_based_method_2(target_compressed_model):")
            print("attack_acc: %.3f%% | attack_auc: %.3f%% |attack_tpr_at_0.001_fpr: %.3f%%" % (c_modentr*100, c_mod_auc*100, c_mod_tpr_at_low_fpr*100))
    
    def CompLeakNR_SAMIA(self):
        if self.original:
            original_attack_predicts = torch.cat([self.attack_original_in_predicts, self.attack_original_out_predicts], dim=0)
            original_attack_sens = torch.cat([self.attack_original_in_sens, self.attack_original_out_sens], dim=0)
            attack_targets = torch.cat([self.attack_in_targets, self.attack_out_targets], dim=0)
            attack_targets = F.one_hot(attack_targets, num_classes=self.num_cls).float()
            attack_labels = torch.cat([torch.ones(self.attack_in_targets.size(0)), torch.zeros(self.attack_out_targets.size(0))], dim=0).long()

            original_victim_predicts = torch.cat([self.victim_original_in_predicts, self.victim_original_out_predicts], dim=0)
            original_victim_sens = torch.cat([self.victim_original_in_sens, self.victim_original_out_sens], dim=0)
            victim_targets = torch.cat([self.victim_in_targets, self.victim_out_targets], dim=0)
            victim_targets = F.one_hot(victim_targets, num_classes=self.num_cls).float()
            victim_labels = torch.cat([torch.ones(self.victim_in_targets.size(0)), torch.zeros(self.victim_out_targets.size(0))], dim=0).long()

            original_new_attack_data = torch.cat([original_attack_predicts, original_attack_sens, attack_targets], dim=1)
            original_new_victim_data = torch.cat([original_victim_predicts, original_victim_sens, victim_targets], dim=1)

            original_attack_train_dataset = TensorDataset(original_new_attack_data, attack_labels)
            original_attack_train_dataloader = DataLoader(original_attack_train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)
            original_attack_test_dataset = TensorDataset(original_new_victim_data, victim_labels)
            original_attack_test_dataloader = DataLoader(original_attack_test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        
            original_attack_model = BaseModel("", "transformer", device=self.device, num_cls=original_new_victim_data.size(1), optimizer="sgd", lr=0.001, weight_decay=5e-4, epochs=100)
            original_best_acc = 0
            original_best_auc = 0
            original_best_tpr = 0
            for epoch in range(100):
                train_acc, train_loss = original_attack_model.train(epoch, original_attack_train_dataloader)
                test_acc, test_loss, auc, tpr_at_low_fpr = original_attack_model.test_SAMIA(original_attack_test_dataloader)
                if test_acc > original_best_acc:
                    original_best_acc = test_acc
                    original_best_auc = auc
                    original_best_tpr_at_low_fpr = tpr_at_low_fpr
            print(f"CompLeakNR_SAMIA(target_original_model):")
            print("attack_acc: %.3f%% | attack_auc: %.3f%% |attack_tpr_at_0.001_fpr: %.3f%%" % (original_best_acc*100, original_best_auc*100, original_best_tpr_at_low_fpr*100))

        if self.compress:
            compress_attack_predicts = torch.cat([self.attack_compress_in_predicts, self.attack_compress_out_predicts], dim=0)
            compress_attack_sens = torch.cat([self.attack_compress_in_sens, self.attack_compress_out_sens], dim=0)
            compress_victim_predicts = torch.cat([self.victim_compress_in_predicts, self.victim_compress_out_predicts], dim=0)
            compress_victim_sens = torch.cat([self.victim_compress_in_sens, self.victim_compress_out_sens], dim=0)
            
            attack_targets = torch.cat([self.attack_in_targets, self.attack_out_targets], dim=0)
            attack_targets = F.one_hot(attack_targets, num_classes=self.num_cls).float()
            attack_labels = torch.cat([torch.ones(self.attack_in_targets.size(0)), torch.zeros(self.attack_out_targets.size(0))], dim=0).long()
            
            victim_targets = torch.cat([self.victim_in_targets, self.victim_out_targets], dim=0)
            victim_targets = F.one_hot(victim_targets, num_classes=self.num_cls).float()
            victim_labels = torch.cat([torch.ones(self.victim_in_targets.size(0)), torch.zeros(self.victim_out_targets.size(0))], dim=0).long()
            compress_new_attack_data = torch.cat([compress_attack_predicts, compress_attack_sens, attack_targets], dim=1)
            compress_new_victim_data = torch.cat([compress_victim_predicts, compress_victim_sens, victim_targets], dim=1)

            compress_attack_train_dataset = TensorDataset(compress_new_attack_data, attack_labels)
            compress_attack_train_dataloader = DataLoader(compress_attack_train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)
            compress_attack_test_dataset = TensorDataset(compress_new_victim_data, victim_labels)
            compress_attack_test_dataloader = DataLoader(compress_attack_test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)
            compress_attack_model = BaseModel("", "transformer", device=self.device, num_cls=compress_new_victim_data.size(1), optimizer="sgd", lr=0.001, weight_decay=5e-4, epochs=100)

            compress_best_acc = 0
            compress_best_auc = 0
            compress_best_tpr = 0
            for epoch in range(100):
                train_acc, train_loss = compress_attack_model.train(epoch, compress_attack_train_dataloader)
                test_acc, test_loss, auc, tpr_at_low_fpr = compress_attack_model.test_SAMIA(compress_attack_test_dataloader)
                if test_acc > compress_best_acc:
                    compress_best_acc = test_acc
                    compress_best_auc = auc
                    compress_best_tpr_at_low_fpr = tpr_at_low_fpr
            print(f"CompLeakNR_SAMIA(target_compressed_model):")
            print("attack_acc: %.3f%% | attack_auc: %.3f%% |attack_tpr_at_0.001_fpr: %.3f%%" % (compress_best_acc*100, compress_best_auc*100, compress_best_tpr_at_low_fpr*100))

        

    def CompLeakNR_RMIA(self):
        a_param: float = 0.5
        gamma: float = 1.0

    
        victim_orginal_model = self.victim_model.model
        victim_orginal_model.eval()
        shadow_num = len(self.shadow_model_list)
        # print(f"Number of shadow models: {shadow_num}")

        
        victim_compress_model = self.victim_compress_model.model
        victim_compress_model.eval()

        shadow_original_models = []
        shadow_compress_models = []

        for shadow_model, shadow_compress_model in zip(self.shadow_model_list, self.shadow_compress_model_list):
            reference_orginal_model = shadow_model.model
            reference_orginal_model.eval()
            shadow_original_models.append(reference_orginal_model)

            reference_compress_model = shadow_compress_model.model
            reference_compress_model.eval()
            shadow_compress_models.append(reference_compress_model)
        
        shadow_original_models = tuple(shadow_original_models)
        shadow_compress_models = tuple(shadow_compress_models)
        
        victim_train_dataset = self.victim_train_loader.dataset
        victim_test_dataset = self.victim_test_loader.dataset
        
        data = ConcatDataset([victim_train_dataset, victim_test_dataset])

        membership_train = torch.ones(len(victim_train_dataset)) 
        membership_test = torch.zeros(len(victim_test_dataset)) 
        data_membership = torch.cat([membership_train, membership_test])

        

        shadow_train_loader = self.shadow_train_loader_list[0]
        shadow_test_loader = self.shadow_test_loader_list[0]

        shadow_train_dataset = shadow_train_loader.dataset
        shadow_test_dataset = shadow_test_loader.dataset

        data_out = ConcatDataset([shadow_train_dataset, shadow_test_dataset])
        z_loader = DataLoader(data_out, batch_size=self.batch_size, shuffle=False)
        data_loader = DataLoader(data, batch_size=self.batch_size, shuffle=False)
        
        threshold = 0.7

        
        if self.original:
            population_ratio_org = np.zeros(len(data_out), dtype=np.float32)
            get_ratio("Z", population_ratio_org, victim_orginal_model, shadow_original_models, z_loader, shadow_num, self.device, "Computing z ratio for original model")
            population_ratio_org.sort()

            ratio_x_org = np.zeros(len(data), dtype=np.float32)
            get_ratio("X", ratio_x_org, victim_orginal_model, shadow_original_models, data_loader, shadow_num, self.device, "Computing x ratio for original model")

            attack_scores_org = []
            for x_r in tqdm(ratio_x_org, desc="Final Score Calculation (Original)", disable=True):
                count = sum(1 for z_r in population_ratio_org if x_r / (z_r + 1e-12) > gamma)
                attack_scores_org.append(count / len(population_ratio_org))
            
            
            
            attack_results_org = np.array(attack_scores_org)
            auc_score_org = roc_auc_score(data_membership, attack_results_org)
            binary_predictions_org = (attack_results_org > threshold).astype(int)
            acc_org = accuracy_score(data_membership, binary_predictions_org)
            tpr_at_low_fpr_org = calculate_tpr_at_fpr(data_membership, attack_results_org, 0.001)

            print(f"CompLeakNR_RMIA(target_original_model):")
            print("attack_acc: %.3f%% | attack_auc: %.3f%% | attack_tpr_at_0.001_fpr: %.3f%%" % 
                (acc_org * 100, auc_score_org * 100, tpr_at_low_fpr_org * 100))
    
   
   
        if self.compress:
            population_ratio_comp = np.zeros(len(data_out), dtype=np.float32)
            get_ratio("Z", population_ratio_comp, victim_compress_model, shadow_compress_models, z_loader, shadow_num, self.attack_device, "Computing z ratio for compressed model")
            population_ratio_comp.sort()

            ratio_x_comp = np.zeros(len(data), dtype=np.float32)
            get_ratio("X", ratio_x_comp, victim_compress_model, shadow_compress_models, data_loader, shadow_num, self.attack_device, "Computing x ratio for compressed model")

            attack_scores_comp = []
            for x_r in tqdm(ratio_x_comp, desc="Final Score Calculation (Compressed)", disable=True):
                count = sum(1 for z_r in population_ratio_comp if x_r / (z_r + 1e-12) > gamma)
                attack_scores_comp.append(count / len(population_ratio_comp))
            
            attack_results_comp = np.array(attack_scores_comp)
            auc_score_comp = roc_auc_score(data_membership, attack_results_comp)
            binary_predictions_comp = (attack_results_comp > threshold).astype(int)
            acc_comp = accuracy_score(data_membership, binary_predictions_comp)
            tpr_at_low_fpr_comp = calculate_tpr_at_fpr(data_membership, attack_results_comp, 0.001)

            print(f"CompLeakNR_RMIA(target_compressed_model):")
            print("attack_acc: %.3f%% | attack_auc: %.3f%% | attack_tpr_at_0.001_fpr: %.3f%%" % 
                (acc_comp * 100, auc_score_comp * 100, tpr_at_low_fpr_comp * 100))


                        