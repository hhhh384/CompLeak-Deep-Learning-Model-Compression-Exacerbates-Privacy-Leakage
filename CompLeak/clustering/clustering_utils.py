import torch
import torch.nn as nn
import numpy as np
import math
from sklearn.cluster import KMeans
import torch.optim as optim
import torch.nn.functional as F
from scipy.sparse import csc_matrix, csr_matrix


class NetCodebook():
    def __init__(self, conv_bits, fc_bits):
        self.conv_bits = conv_bits
        self.fc_bits = fc_bits
        self.codebook_index = []
        self.codebook_value = []

    def add_layer_codebook(self, layer_codebook_index, layer_codebook_value):
        self.codebook_index.append(layer_codebook_index)
        self.codebook_value.append(layer_codebook_value)

'''
quantization_conv_bits = 8
quantization_fc_bits = 4

conv_layer_length, codebook, nz_num = share_weight(victim_cluster_model.model, quantization_conv_bits, quantization_fc_bits)


def share_weight(net, conv_bits, fc_bits):
    conv_layer_num, fc_layer_num, nz_num, conv_value_array, fc_value_array, layer_types = load_model(net)
    conv_index = 0
    fc_index = 0
    #codebook = NetCodebook(conv_bits, fc_bits)
    for i in range(len(nz_num)):
        layer_type = layer_types[i]  
        if layer_type == 'fc':
            bits = fc_bits
            layer_weight = fc_value_array[fc_index:fc_index + nz_num[i]]
            fc_index += nz_num[i]
        else:
            bits = conv_bits
            layer_weight = conv_value_array[conv_index:conv_index + nz_num[i]]
            conv_index += nz_num[i]
        min_weight = min(layer_weight)
        max_weight = max(layer_weight)
        n_clusters = 2 ** bits
        space = np.linspace(min_weight, max_weight, num=n_clusters, dtype=np.float32)
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=1, precompute_distances=True, algorithm="full")
        kmeans.fit(layer_weight.reshape(-1, 1))
        codebook_index = np.array(kmeans.labels_, dtype=np.uint8)
        codebook_value = kmeans.cluster_centers_[:n_clusters]
        codebook.add_layer_codebook(codebook_index.reshape(-1), codebook_value.reshape(-1))
    return conv_layer_num, codebook, nz_num
'''

def apply_weight_sharing(model, bits=5):
    for module in model.children():
        dev = module.weight.device
        weight = module.weight.data.cpu().numpy()
        shape = weight.shape
        mat = csr_matrix(weight) if shape[0] < shape[1] else csc_matrix(weight)
        min_ = min(mat.data)
        max_ = max(mat.data)
        space = np.linspace(min_, max_, num=2**bits)
        kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1,1), n_init=1, precompute_distances=True, algorithm="full")
        kmeans.fit(mat.data.reshape(-1,1))
        new_weight = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)
        mat.data = new_weight
        module.weight.data = torch.from_numpy(mat.toarray()).to(dev)


'''
def share_weight(net, bits=4):
    for module in net.children():
        dev = module.weight.device
        weight = module.weight.data.cpu().numpy()
        # shape = weight.shape
        # mat = csr_matrix(weight) if shape[0] < shape[1] else csc_matrix(weight)
        min_ = min(weight.data)
        max_ = max(weight.data)
        n_clusters = 2 ** bits
        space = np.linspace(min_, max_, num=n_clusters, dtype=np.float32)
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=1, precompute_distances=True, algorithm="full")
        kmeans.fit(weight.reshape(-1, 1))
        new_weight = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)
        weight.data = new_weight
        module.weight.data = torch.from_numpy(weight.toarray()).to(dev)
        
        #codebook_index = np.array(kmeans.labels_, dtype=np.uint8)
        #codebook_value = kmeans.cluster_centers_[:n_clusters]
        #codebook.add_layer_codebook(codebook_index.reshape(-1), codebook_value.reshape(-1))
       
    return net
'''

def load_model(net):
    ''' 
        参数:
            net (torch.nn.Module): 训练好的普通模型
        
        返回:
            conv_layer_num (int): 卷积层数量
            fc_layer_num (int): 全连接层数量
            nz_num (list): 每层非零元素的数量
            conv_value_array (np.ndarray): 卷积层非零元素的值
            fc_value_array (np.ndarray): 全连接层非零元素的值
            layer_types (list): 每层的类型列表（'conv' 或 'fc'）
    '''
    conv_layer_num = 0
    fc_layer_num = 0
    nz_num = []
    conv_value_array = []
    fc_value_array = []
    layer_types = []
    
    for name, param in net.named_parameters():
        if 'conv' in name:
            conv_layer_num += 1
            conv_weight = param.data.numpy()
            non_zero_values = conv_weight[conv_weight != 0]    
            nz_num.append(len(non_zero_values))
            conv_value_array.extend(non_zero_values)
            layer_types.append('conv')
        elif 'fc' in name:
            fc_layer_num += 1
            fc_weight = param.data.numpy()
            non_zero_values = fc_weight[fc_weight != 0]
            nz_num.append(len(non_zero_values))
            fc_value_array.extend(non_zero_values)
            layer_types.append('fc')

    conv_value_array = np.array(conv_value_array, dtype=np.float32)
    fc_value_array = np.array(fc_value_array, dtype=np.float32)

    return conv_layer_num, fc_layer_num, nz_num, conv_value_array, fc_value_array, layer_types


def share_weight(net, bits=4):
    for module in net.modules():  
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            print(f"Processing {module.__class__.__name__}")
            dev = module.weight.device  # 获取权重所在的设备
            weight = module.weight.data.cpu().numpy()  # 将权重转成NumPy数组
            min_ = np.min(weight)  # 找到最小值
            max_ = np.max(weight)  # 找到最大值
            n_clusters = 2 ** bits
            space = np.linspace(min_, max_, num=n_clusters, dtype=np.float32)
            kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=1, algorithm="lloyd")
            kmeans.fit(weight.reshape(-1, 1))
            new_weight = kmeans.cluster_centers_[kmeans.labels_].reshape(weight.shape)  # 获取离散化后的权重
            module.weight.data = torch.from_numpy(new_weight).to(dev)  # 将离散化后的权重更新到模型中

    return net

    
