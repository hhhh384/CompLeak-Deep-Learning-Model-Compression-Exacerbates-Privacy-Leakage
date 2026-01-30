import copy
import numpy as np
import torch
import random
import os
import torch.nn.functional as F
from attack.attack_model import DT, RF, LR, MLP
import pandas as pd
from tqdm import tqdm


class ConstructFeature:
    def __init__(self, posterior_df):
        self.posterior_df = posterior_df
    def obtain_feature(self, method, post_df):
        post_df = copy.deepcopy(post_df)
        self.posterior_df[method] = ""
        
        if method == 'CompLeakNR_train_based_method_2(target_original_model)':
            original_list = post_df.original.tolist()
            targets_list = post_df.targets.tolist()
            combined_data = [torch.cat(( original, target), dim=1) for  original, target in zip(original_list, targets_list)]
            combined_series = pd.Series(combined_data)
            self.posterior_df[method] = combined_series

        elif method == 'CompLeakNR_train_based_method_1(target_original_model)':
            self.posterior_df[method] = self.posterior_df.original

        elif method == 'CompLeakNR_train_based_method_1(target_compressed_model)':
            self.posterior_df[method] = self.posterior_df.compress

        elif method == 'CompLeakNR_train_based_method_2(target_compressed_model)':
            compress_list = post_df.compress.tolist()
            targets_list = post_df.targets.tolist()
            combined_data = [torch.cat((compress, target), dim=1) for  compress, target in zip(compress_list, targets_list)]
            combined_series = pd.Series(combined_data)
            self.posterior_df[method] = combined_series

        elif method == 'CompLeakSR1':
            conc_list = []
            for index, posterior in enumerate(post_df.original):
                sort_indices = np.argsort(posterior[0, :]).numpy()
                original_posterior = posterior[0, sort_indices].reshape(1, -1)
                compress_posterior = post_df.compress[index][0, sort_indices].reshape(1, -1)
                conc = np.concatenate(
                    (original_posterior, compress_posterior),
                    axis=1
                )
                conc_list.append(conc)
            self.posterior_df[method] = pd.Series(conc_list, index=post_df.index)

        elif method == 'CompLeakSR2':
            conc_list = []
            for index, posterior in enumerate(post_df.original):
                sort_indices = np.argsort(posterior[0, :]).numpy()

                original_posterior = posterior[0, sort_indices].reshape(1, -1)
                compress_posterior = post_df.compress[index][0, sort_indices].reshape(1, -1)
                target = post_df.targets[index][0, sort_indices].reshape(1, -1)

                conc = np.concatenate(
                    (original_posterior, compress_posterior, target),
                    axis=1
                )
                conc_list.append(conc)
            self.posterior_df[method] = pd.Series(conc_list, index=post_df.index)

        else:
            raise ValueError(f"Invalid feature construction method: {method}")


class Attack:
    def __init__(self, attack_model_name, shadow_post_df, target_post_df):

        self.attack_model_name = attack_model_name
        self.shadow_post_df = shadow_post_df
        self.target_post_df = target_post_df

        self.attack_model = self.determine_attack_model(attack_model_name)

    def determine_attack_model(self, attack_model_name):
        if attack_model_name == 'LR':
            return LR()
        elif attack_model_name == 'DT':
            return DT()
        elif attack_model_name == 'RF':
            return RF()
        elif attack_model_name == 'MLP':
            return MLP()
        else:
            raise Exception("invalid attack name")

    def train_attack_model(self, feature_construct_method, save_path):
        seed_n = 42
        g = torch.Generator()
        g.manual_seed(seed_n)
        random.seed(seed_n)
        np.random.seed(seed_n)
        torch.manual_seed(seed_n)
        torch.cuda.manual_seed(seed_n)
        torch.cuda.manual_seed_all(seed_n)
        os.environ['PYTHONHASHSEED'] = str(seed_n) 
        
        self.shadow_feature = self._concatenate_feature(self.shadow_post_df, feature_construct_method)
        label = self.shadow_post_df.label.astype('int')
    
        self.attack_model.train_model(self.shadow_feature, label, save_name = None)

        train_acc, _, prob = self.attack_model.test_model_acc(self.shadow_feature, label)
        train_auc, train_tpr_at_low_fpr = self.attack_model.test_model_auc(self.shadow_feature, label)
        # print("attack model (%s, %s): train_acc:  %.3f | train_auc:  %.3f |train_tpr_at_0.001_fpr:  %.3f" % (self.attack_model_name, feature_construct_method, train_acc, train_auc, train_tpr_at_low_fpr))
        return train_acc, train_auc, prob, train_tpr_at_low_fpr

    def test_attack_model(self, feature_construct_method):
        self.target_feature = self._concatenate_feature(self.target_post_df, feature_construct_method)
        label = self.target_post_df.label.astype('int')

        test_acc, pred, prob = self.attack_model.test_model_acc(self.target_feature, label)
        test_auc, test_tpr_at_low_fpr = self.attack_model.test_model_auc(self.target_feature, label)
        print("attack model (%s): attack_acc: %.3f%% | attack_auc: %.3f%% |attack_tpr_at_0.001_fpr: %.3f%%" % (self.attack_model_name, test_acc*100, test_auc*100, test_tpr_at_low_fpr))
        return test_acc, test_auc, pred, prob, test_tpr_at_low_fpr


    def _concatenate_feature(self, posterior, method):
        feature = np.zeros((posterior[method][0].shape))
        for _, post in enumerate(posterior[method]):
            feature = np.concatenate((feature, post), axis=0)
        return feature[1:, :]
    

"""
This code is modified from https://github.com/inspire-group/membership-inference-evaluation
"""
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.interpolate import interp1d
from scipy.special import softmax


class ThresholdAttacker:
    def __init__(self, shadow_train_performance, shadow_test_performance,  target_train_performance,
                 target_test_performance, num_classes):
        self.num_classes = num_classes

        self.s_tr_outputs, self.s_tr_labels = shadow_train_performance
        self.s_te_outputs, self.s_te_labels = shadow_test_performance
        self.t_tr_outputs, self.t_tr_labels = target_train_performance
        self.t_te_outputs, self.t_te_labels = target_test_performance

        self.s_tr_labels = self.s_tr_labels.astype(int)
        self.s_te_labels = self.s_te_labels.astype(int)
        self.t_tr_labels = self.t_tr_labels.astype(int)
        self.t_te_labels = self.t_te_labels.astype(int)


        self.s_tr_corr = (np.argmax(self.s_tr_outputs, axis=1) == self.s_tr_labels).astype(int)
        self.s_te_corr = (np.argmax(self.s_te_outputs, axis=1) == self.s_te_labels).astype(int)
        self.t_tr_corr = (np.argmax(self.t_tr_outputs, axis=1) == self.t_tr_labels).astype(int)
        self.t_te_corr = (np.argmax(self.t_te_outputs, axis=1) == self.t_te_labels).astype(int)

    

        self.s_tr_conf = np.array([self.s_tr_outputs[i, self.s_tr_labels[i]] for i in range(len(self.s_tr_labels))])
        self.s_te_conf = np.array([self.s_te_outputs[i, self.s_te_labels[i]] for i in range(len(self.s_te_labels))])
        self.t_tr_conf = np.array([self.t_tr_outputs[i, self.t_tr_labels[i]] for i in range(len(self.t_tr_labels))])
        self.t_te_conf = np.array([self.t_te_outputs[i, self.t_te_labels[i]] for i in range(len(self.t_te_labels))])

        self.s_tr_entr = self._entr_comp(self.s_tr_outputs)
        self.s_te_entr = self._entr_comp(self.s_te_outputs)
        self.t_tr_entr = self._entr_comp(self.t_tr_outputs)
        self.t_te_entr = self._entr_comp(self.t_te_outputs)

        self.s_tr_m_entr = self._m_entr_comp(self.s_tr_outputs, self.s_tr_labels)
        self.s_te_m_entr = self._m_entr_comp(self.s_te_outputs, self.s_te_labels)
        self.t_tr_m_entr = self._m_entr_comp(self.t_tr_outputs, self.t_tr_labels)
        self.t_te_m_entr = self._m_entr_comp(self.t_te_outputs, self.t_te_labels)

    def _log_value(self, probs, small_value=1e-30):
        return -np.log(np.maximum(probs, small_value))

    def _entr_comp(self, probs):
        return np.sum(np.multiply(probs, self._log_value(probs)), axis=1)

    def _m_entr_comp(self, probs, true_labels):
        log_probs = self._log_value(probs)
        reverse_probs = 1 - probs
        log_reverse_probs = self._log_value(reverse_probs)
        modified_probs = np.copy(probs)
        modified_probs[range(true_labels.size), true_labels] = reverse_probs[range(true_labels.size), true_labels]
        modified_log_probs = np.copy(log_reverse_probs)
        modified_log_probs[range(true_labels.size), true_labels] = log_probs[range(true_labels.size), true_labels]
        return np.sum(np.multiply(modified_probs, modified_log_probs), axis=1)

    def _thre_setting(self, tr_values, te_values):
        value_list = np.concatenate((tr_values, te_values))
        thre, max_acc = 0, 0
        for value in value_list:
            tr_ratio = np.sum(tr_values >= value) / (len(tr_values) + 0.0)
            te_ratio = np.sum(te_values < value) / (len(te_values) + 0.0)
            acc = 0.5 * (tr_ratio + te_ratio)
            if acc > max_acc:
                thre, max_acc = value, acc
        return thre

    def compute_auc_tpr(self, ground_truth, predicted_probs):
        """计算 AUC 和 TPR@0.1% FPR"""
        auc_score = roc_auc_score(ground_truth, predicted_probs)

        fpr, tpr, thresholds = roc_curve(ground_truth, predicted_probs)
        fpr_target = 0.001  # 0.1% FPR
        interp_func = interp1d(fpr, tpr)
        tpr_0 = interp_func(fpr_target)

        return auc_score, tpr_0



    def _mem_inf_thre(self, v_name, s_tr_values, s_te_values, t_tr_values, t_te_values):
        # perform membership inference attack by thresholding feature values: the feature can be prediction confidence,
        # (negative) prediction entropy, and (negative) modified entropy
        t_tr_mem, t_te_non_mem = 0, 0
        for num in range(self.num_classes):
            thre = self._thre_setting(s_tr_values[self.s_tr_labels == num], s_te_values[self.s_te_labels == num])
            t_tr_mem_tmp = np.sum(t_tr_values[self.t_tr_labels == num] >= thre)
            t_tr_mem += t_tr_mem_tmp
            t_te_non_mem_tmp = np.sum(t_te_values[self.t_te_labels == num] < thre)
            t_te_non_mem += t_te_non_mem_tmp
            tmp_acc = 0.5 * (t_tr_mem_tmp / (len(t_tr_values[self.t_tr_labels == num]) + 0.0) +
                            t_te_non_mem_tmp / (len(t_te_values[self.t_te_labels == num]) + 0.0))
    
        if t_tr_values.ndim == 1:  
            t_tr_probs = 1 / (1 + np.exp(-t_tr_values))  
            t_te_probs = 1 / (1 + np.exp(-t_te_values))
        else:  
            t_tr_probs = softmax(t_tr_values, axis=1)
            t_te_probs = softmax(t_te_values, axis=1)

        if t_tr_probs.ndim == 2:
            t_tr_probs_pos = t_tr_probs[:, 1]  
            t_te_probs_pos = t_te_probs[:, 1]  
        else:
            t_tr_probs_pos = t_tr_probs
            t_te_probs_pos = t_te_probs
        all_ground_truth = np.concatenate((np.ones(len(t_tr_probs_pos)), np.zeros(len(t_te_probs_pos))))
        all_pred_probs = np.concatenate((t_tr_probs_pos, t_te_probs_pos))
        auc1, tpr_0 = self.compute_auc_tpr(all_ground_truth, all_pred_probs)
        mem_inf_acc = 0.5 * (t_tr_mem / (len(self.t_tr_labels) + 0.0) + t_te_non_mem / (len(self.t_te_labels) + 0.0))
        return mem_inf_acc, auc1, tpr_0

    

    def _mem_inf_thre_non_cls(self, v_name, s_tr_values, s_te_values, t_tr_values, t_te_values):
    # perform membership inference attack by thresholding feature values: the feature can be prediction confidence,
    # (negative) prediction entropy, and (negative) modified entropy
        thre = self._thre_setting(s_tr_values, s_te_values)
        t_tr_mem = np.sum(t_tr_values >= thre)
        t_te_non_mem = np.sum(t_te_values < thre)
        mem_inf_acc = 0.5 * (t_tr_mem / (len(self.t_tr_labels) + 0.0) + t_te_non_mem / (len(self.t_te_labels) + 0.0))

    
  
        if t_tr_values.ndim == 1:  
            t_tr_probs = 1 / (1 + np.exp(-t_tr_values))  
            t_te_probs = 1 / (1 + np.exp(-t_te_values))
        else:  
            t_tr_probs = softmax(t_tr_values, axis=1)
            t_te_probs = softmax(t_te_values, axis=1)

        if t_tr_probs.ndim == 2:
            t_tr_probs_pos = t_tr_probs[:, 1]  
            t_te_probs_pos = t_te_probs[:, 1]  
        else:
            t_tr_probs_pos = t_tr_probs
            t_te_probs_pos = t_te_probs
        all_ground_truth = np.concatenate((np.ones(len(t_tr_probs_pos)), np.zeros(len(t_te_probs_pos))))
        all_pred_probs = np.concatenate((t_tr_probs_pos, t_te_probs_pos))

 
        auc1, tpr_0 = self.compute_auc_tpr(all_ground_truth, all_pred_probs)


        print(f'Membership Inference Accuracy: {mem_inf_acc:.4f}, AUC: {auc1:.4f}, TPR@0.1% FPR: {tpr_0:.4f}')
        
        return mem_inf_acc, auc1, tpr_0


    def _mem_inf_benchmarks(self):
        confidence, con_auc, con_tpr = \
            self._mem_inf_thre('confidence', self.s_tr_conf, self.s_te_conf, self.t_tr_conf, self.t_te_conf)
        entropy, ent_auc, ent_tpr = \
            self._mem_inf_thre('entropy', -self.s_tr_entr, -self.s_te_entr, -self.t_tr_entr, -self.t_te_entr)
        modentr, mod_auc, mod_tpr = \
            self._mem_inf_thre('modified entropy', -self.s_tr_m_entr, -self.s_te_m_entr, -self.t_tr_m_entr,
                               -self.t_te_m_entr)
        return confidence,con_auc, con_tpr, entropy, ent_auc, ent_tpr, modentr, mod_auc, mod_tpr

    def _mem_inf_benchmarks_non_cls(self):
        confidence,con_auc, con_tpr = \
            self._mem_inf_thre_non_cls('confidence', self.s_tr_conf, self.s_te_conf, self.t_tr_conf, self.t_te_conf)
        entropy, ent_auc, ent_tpr = \
            self._mem_inf_thre_non_cls('entropy', -self.s_tr_entr, -self.s_te_entr, -self.t_tr_entr, -self.t_te_entr)
        modentr, mod_auc, mod_tpr  = \
            self._mem_inf_thre_non_cls('modified entropy', -self.s_tr_m_entr, -self.s_te_m_entr, -self.t_tr_m_entr,
                               -self.t_te_m_entr)
        return confidence,con_auc, con_tpr, entropy, ent_auc, ent_tpr, modentr, mod_auc, mod_tpr




def get_ratio(type, ratio, model, shadow_models, loader, shadow_num, device, description):
    ptr = 0
    a_param: float = 0.5
    gamma: float = 1.0

    with torch.no_grad():
        for imgs, labels in tqdm(loader, total=len(loader), desc=description, disable=True):
            imgs, labels = imgs.to(device), labels.to(device)
            B = imgs.size(0)

            # Compute Pr(. | \theta) for the target model.
            target = F.softmax(model(imgs), dim=1) # [B, num_classes]
            conf = target[range(B), labels].cpu().numpy() # [B]

            # Compute shadow model confidences.
            shadow_conf = np.zeros((shadow_num, B), dtype=np.float32) # [num_shadow_models, B]
            for idx, sm in enumerate(shadow_models):
                probs_sm = F.softmax(sm(imgs), dim=1) # [B, num_classes]
                conf_sm = probs_sm[range(B), labels].cpu().numpy() # [B]
                shadow_conf[idx] = conf_sm # [num_shadow_models, B]

            pr_out = shadow_conf.mean(axis=0) # [B]

            if type == "X":
                # Apply linear correction for target samples.
                pr = 0.5 * ((1.0 + a_param) * pr_out + (1.0 - a_param))
            elif type == "Z":
                pr = pr_out
            else:
                error_msg = f"Invalid type '{type}' provided to get_ratio. Expected 'X' or 'Z'."    
                raise ValueError(error_msg)

            # Compute ratio = Pr(. | θ) / Pr(.)
            ratio[ptr: ptr+B] = conf / np.maximum(pr, 1e-12)

            ptr += B
            
            
def compute_quantiles(sorted_arr, measures):
    """
    Compute the quantile values for a set of measures based on a sorted reference array.

    Args:
        sorted_arr (numpy.ndarray or list): A sorted array that serves as the reference
            distribution. This array must be sorted in ascending order.
        measures (numpy.ndarray or list): An array of values for which to compute 
            quantile scores based on the reference array.

    Returns:
        numpy.ndarray: An array of quantile scores, where each value is between 0 and 1, 
        representing the proportion of elements in `sorted_arr` that are less than or 
        equal to the corresponding element in `measures`.
    """
    indices = np.searchsorted(sorted_arr, measures, side='right')
    return indices / len(sorted_arr)



def calculate_tpr_at_fpr(y_true, y_score, target_fpr):
    members_scores = y_score[y_true == 1]
    non_members_scores = y_score[y_true == 0]
    percentile = 100 * (1 - target_fpr)
    threshold = np.percentile(non_members_scores, percentile)
    true_positives = np.sum(members_scores > threshold)
    tpr = true_positives / len(members_scores)
    return tpr