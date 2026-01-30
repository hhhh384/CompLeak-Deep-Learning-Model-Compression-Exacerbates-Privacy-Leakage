import argparse
import json
import numpy as np
import pickle
import random
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import ConcatDataset, DataLoader, Subset
from base_model import BaseModel
from data_utils import get_dataset
from attack.attackers import MiaAttack


parser = argparse.ArgumentParser()
parser.add_argument('--device', default=0, type=int, help="GPU id to use")
parser.add_argument('--dataset_name', default='mini_imagenet', type=str)
parser.add_argument('--model_name', default='vgg16', type=str)
parser.add_argument('--num_cls', default=100, type=int)

parser.add_argument('--input_dim', default=3, type=int)
parser.add_argument('--seed', default=7, type=int)
parser.add_argument('--batch_size', default=32, type=int)

parser.add_argument('--original', action='store_true')
parser.add_argument('--compress', action='store_true')
parser.add_argument('--compress_name', default="pruning", type=str)
parser.add_argument('--compress_degree', default="0.8", type=str)
parser.add_argument('--shadow_num', default=1, type=int)



def main(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = f"cuda:{args.device}"
    cpu_device = torch.device("cpu:0")
    cudnn.benchmark = True

    if args.compress_name == "quantization":
         attack_device = cpu_device
    else:
        attack_device = device
    print(f"attack_device:{ attack_device}")

    save_folder_original = f"results/{args.dataset_name}_{args.model_name}"
    save_folder_compress = f"results_{args.compress_name}/{args.dataset_name}_{args.model_name}_{args.compress_degree}"
    
    trainset = get_dataset(args.dataset_name, train=True)
    testset = get_dataset(args.dataset_name, train=False)
    
    data_path = f"data/{args.dataset_name}_data_index_inference.pkl"
    
    with open(data_path, 'rb') as f:
        inference_victim_train_list, inference_victim_test_list, inference_attack_split_list = pickle.load(f)
    victim_train_dataset = Subset(trainset, inference_victim_train_list)
    victim_test_dataset = Subset(testset, inference_victim_test_list)
    
    victim_train_loader = DataLoader(victim_train_dataset, batch_size=args.batch_size, shuffle=False)
    victim_test_loader = DataLoader(victim_test_dataset, batch_size=args.batch_size, shuffle=False)
    

    # Load victim model
    victim_model_save_folder = save_folder_original + "/victim_model"
    victim_model_path = f"{victim_model_save_folder}/best.pth"
    victim_model = BaseModel(args.dataset_name, args.model_name, num_cls=args.num_cls, input_dim=args.input_dim, device=device)
    victim_model.load(victim_model_path)


    compress_victim_model_save_folder = save_folder_compress + "/victim_model"
    compress_victim_model_path = f"{compress_victim_model_save_folder}/best.pth"
    print(f"Load Compress Model from {compress_victim_model_save_folder}")
    victim_compress_model = BaseModel(args.dataset_name, args.model_name, num_cls=args.num_cls, input_dim=args.input_dim, device=attack_device)
    if args.compress_name == "quantization":
        victim_compress_model.load_torchscript_model(compress_victim_model_path)
    elif args.compress_name == "clustering":
        victim_compress_model.load(f"{compress_victim_model_save_folder}/best.pth")
    elif args.compress_name == "pruning":
        victim_compress_model.model.load_state_dict(torch.load(f"{compress_victim_model_save_folder}/best.pth"))



    # Load shadow models
    shadow_model_list, shadow_compress_model_list, shadow_train_loader_list, shadow_test_loader_list = [], [], [], []
    for shadow_ind in range(args.shadow_num):
        
        attack_train_list, attack_test_list = inference_attack_split_list[shadow_ind]   
        shadow_train_dataset = Subset(trainset, attack_train_list)
        shadow_test_dataset = Subset(testset, attack_test_list)

        shadow_train_loader = DataLoader(shadow_train_dataset, batch_size=args.batch_size, shuffle=False)
        shadow_test_loader = DataLoader(shadow_test_dataset, batch_size=args.batch_size, shuffle=False)
        
        shadow_model_path = f"{save_folder_original}/shadow_model_{shadow_ind}/best.pth"
        shadow_model = BaseModel(args.dataset_name, args.model_name, num_cls=args.num_cls, input_dim=args.input_dim, device=device)
        shadow_model.load(shadow_model_path)


        compress_shadow_model_save_folder = f"{save_folder_compress}/shadow_model_{shadow_ind}"
        compress_shadow_model_path = f"{compress_shadow_model_save_folder}/best.pth"
        print(f"Load Compress Shadow Model From {compress_shadow_model_save_folder}")
        shadow_compress_model = BaseModel(args.dataset_name, args.model_name, num_cls=args.num_cls, input_dim=args.input_dim, device= attack_device)
        if args.compress_name == "quantization":
            shadow_compress_model.load_torchscript_model(compress_shadow_model_path)
        elif args.compress_name == "clustering":
            shadow_compress_model.load(compress_shadow_model_path)
        elif args.compress_name == "pruning":
            shadow_compress_model.model.load_state_dict(torch.load(f"{compress_shadow_model_save_folder}/best.pth"))
        

        shadow_model_list.append(shadow_model)
        shadow_compress_model_list.append(shadow_compress_model)
        shadow_train_loader_list.append(shadow_train_loader)
        shadow_test_loader_list.append(shadow_test_loader)


    print("Start Membership Inference Attacks")
    attacker = MiaAttack(
        victim_model, victim_compress_model, victim_train_loader, victim_test_loader,
        shadow_model_list, shadow_compress_model_list, shadow_train_loader_list, shadow_test_loader_list,
         original = args.original,  compress = args.compress, num_cls=args.num_cls, device = device, attack_device=attack_device,  batch_size=args.batch_size, save_folder_compress=save_folder_compress)

    attacker.CompLeakSR()


if __name__ == '__main__':
    args = parser.parse_args()

    print(args)
    main(args)
