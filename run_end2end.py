from transformers import OPTForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from torch import nn
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
)
from utils import Collator, Huggingface_dataset, ExponentialMovingAverage

from obc_solver_func import (
    reconstruct_best_weight, 
    reconstruct_best_weight_xty,
    prune_unstructured,
    prune_structured,
    bias_and_weights,
    restore_layer_weights
)

def load_data(tokenizer):
    # dataloader
    collator = Collator(pad_token_id=tokenizer.pad_token_id)
    # for training and dev
    train_dataset = Huggingface_dataset(tokenizer, name_or_dataset="glue", subset="sst2", split="train")      
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, collate_fn=collator)
    dev_dataset = Huggingface_dataset(tokenizer, name_or_dataset="glue",
                                                subset="sst2", split='validation')
    dev_loader = DataLoader(dev_dataset, batch_size=256, shuffle=False, collate_fn=collator)
    test_loader = dev_loader

    return train_dataset, dev_dataset, train_loader, dev_loader, test_loader
        
def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    avg_loss = ExponentialMovingAverage()
    with torch.no_grad():
        for model_inputs, labels in data_loader:
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
            labels = labels.to(device)
            logits = model(**model_inputs).logits
            _, preds = logits.max(dim=-1)
            loss = F.cross_entropy(logits, labels.squeeze(-1))
            avg_loss.update(loss.item())
            correct += (preds == labels.squeeze(-1)).sum().item()
            total += labels.size(0)
        accuracy = correct / (total + 1e-13)
    return accuracy, avg_loss.get_metric()

def print_model_sparsity(network):
    for name, layer in network.named_modules():
        if type(layer) == torch.nn.Linear:
            print(name, 1 - np.mean(np.abs(layer.weight.data.cpu().numpy()) > 1e-10))
                               
def restore_model_weights(network, pruned_weights):
    from copy import deepcopy
    embedding = deepcopy(network.get_input_embeddings())
    if pruned_weights is not None:
        for name, layer in network.named_modules():            
            if name not in pruned_weights or name in ('model.decoder.project_in'):
                continue
            if np.sum(np.isnan(pruned_weights[name])) > 0:
                print(f'pruned result of layer {name} contains nan. Skip.')
                continue
            if type(layer) == torch.nn.Linear:
                # because torch's weight is transpose of tf's weight
                if layer.bias is not None:
                    layer.weight.data = torch.from_numpy(
                        pruned_weights[name][:-1, :].reshape(layer.weight.data.shape[::-1])).float().T.to(network.device)
                    layer.bias.data = torch.from_numpy(pruned_weights[name][-1, :]).float().to(network.device)
                else:
                    layer.weight.data = torch.from_numpy(
                        pruned_weights[name].reshape(layer.weight.data.shape[::-1])).float().T.to(network.device)
    network.model.decoder.embed_tokens = embedding
                    
def add_argument():
    parser=argparse.ArgumentParser(description='Adaptive Package')
    parser.add_argument('--sparsity', type=float, default=0.5,
                    help='sparsity')
    parser.add_argument('--teacher', type=str, default=None,
                    help='teacher file')
    parser.add_argument('--output', type=str, default=None,
                    help='output file')
    args=parser.parse_args()
    return args

def main():
    args = add_argument()
    print("Sparsity:", args.sparsity)

    import pickle
    teacher = args.teacher

    device = "cuda:0"
    # teacher = ""
    outputs = {}
    def create_hook(name):
        def hook(model: torch.nn.Linear, input, output):
            outputs[name] = output.cpu()
        return hook

    model = AutoModelForSequenceClassification.from_pretrained("/hdd1/jianwei/workspace/robust_ticket_soups/dense/outputs2/finetune_glue-sst2_lr2e-05_epochs30_seed426_time1680669486438/epoch19").to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("/hdd1/jianwei/workspace/robust_ticket_soups/dense/outputs2/finetune_glue-sst2_lr2e-05_epochs30_seed426_time1680669486438/epoch19")

    if teacher:
        with open(teacher, 'rb') as handle:
            updated_weights = pickle.load(handle)
        restore_model_weights(model, updated_weights)
    
    handlers={}
    for name, layer in model.named_modules():
        if type(layer) == torch.nn.Linear:
            handlers[name] = layer.register_forward_hook(create_hook(name))

    train_dataset, dev_dataset, train_loader, dev_loader, test_loader = load_data(tokenizer=tokenizer)
    # acc, loss = evaluate(model, dev_loader, device)
    # print("Accuracy:", acc, "Loss:", loss)
    # exit()

    model_inputs = next(iter(train_loader))[0]
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    model(**model_inputs)
    ########################################################################

    del model
    model = AutoModelForSequenceClassification.from_pretrained("/hdd1/jianwei/workspace/robust_ticket_soups/dense/outputs2/finetune_glue-sst2_lr2e-05_epochs30_seed426_time1680669486438/epoch19").to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("/hdd1/jianwei/workspace/robust_ticket_soups/dense/outputs2/finetune_glue-sst2_lr2e-05_epochs30_seed426_time1680669486438/epoch19")
    
    if teacher:
        with open(teacher, 'rb') as handle:
            updated_weights = pickle.load(handle)
        restore_model_weights(model, updated_weights)

    mse = nn.MSELoss()
    pruned_weight = {}
    def create_hook2(name):
        def hook2(module: torch.nn.Linear, input, output):
            # x and xtx
            if module.bias is not None:
                x = torch.cat((input[0], torch.ones(*(*input[0].shape[:-1], 1), device=input[0].device)), dim=-1)
            else:
                x = input[0]
            x = x.view(-1, x.shape[-1])
            y = outputs[name].to(device)

            # for ridge_regular in [1e-4, 1e-3, 1e-2, 1e-1, 1]:
            # w_star
            xtx = torch.matmul(x.T, x).cuda()
            
            if "pooler" not in name and "classifier" not in name:
                w_star = reconstruct_best_weight(xtx, x, y.view(-1, y.shape[-1]), w=bias_and_weights(module), ridge_regular=1e-4)
                # xty = torch.matmul(x.T, y.view(-1, y.shape[-1])).cuda()
                # w_star = reconstruct_best_weight_xty(xtx, xty, w=bias_and_weights(module).cuda(), ridge_regular=1e-4)
                
                # pruning
                w_sparse = prune_unstructured(
                    W=w_star.T,
                    H=xtx,
                    sparsity=args.sparsity
                ).T
                del w_star
            else:
                # xty = torch.matmul(x.T, y.view(-1, y.shape[-1])).cuda()
                # w_sparse = reconstruct_best_weight_xty(xtx, xty, w=bias_and_weights(module), ridge_regular=1e-4)
                w_sparse = reconstruct_best_weight(xtx, x, y.view(-1, y.shape[-1]), w=bias_and_weights(module), ridge_regular=1e-4)

            del xtx
            pruned_weight[name] = w_sparse.cpu().numpy()

            # recalculate output
            y_sparse = torch.matmul(x, w_sparse).view(y.shape)
            error = mse(y.cpu(), y_sparse.cpu())
            print(name, error.item())

                # if error.item() < 1:
                #     break
            
            # update weights and output
            restore_layer_weights(module, w_sparse)
            return y_sparse
          
        return hook2

    handlers={}
    for name, layer in model.named_modules():
        # if type(layer) == torch.nn.Linear and "pooler" not in name and "classifier" not in name:
        if type(layer) == torch.nn.Linear:
            handlers[name] = layer.register_forward_hook(create_hook2(name))

    with torch.no_grad():
        model(**model_inputs)
    ########################################################################
    print_model_sparsity(model)
    
    for _,handler in handlers.items():
        handler.remove()
        
    acc, loss = evaluate(model, dev_loader, device)
    print("Accuracy:", acc, "Loss:", loss)

    if args.output:
        model.save_pretrained(args.output)
        tokenizer.save_pretrained(args.output)

if __name__ == '__main__':
    main()


