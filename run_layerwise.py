import torch
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
    find_layers,
    reconstruct_best_weight, 
    reconstruct_best_weight_xty,
    prune_unstructured,
    prune_structured,
    prune_nmstructured,
    bias_and_weights,
    restore_layer_weights
)

def load_data(tokenizer, batch_size=128):
    # dataloader
    collator = Collator(pad_token_id=tokenizer.pad_token_id)
    # for training and dev
    train_dataset = Huggingface_dataset(tokenizer, name_or_dataset="glue", subset="sst2", split="train")      
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)
    dev_dataset = Huggingface_dataset(tokenizer, name_or_dataset="glue",
                                                subset="sst2", split='validation')
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
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
    parser.add_argument('--model', type=str, default="",
                    help='model path')
    parser.add_argument('--teacher', type=str, default=None,
                    help='teacher file')
    parser.add_argument('--output', type=str, default=None,
                    help='output file')
    args=parser.parse_args()
    return args

def main():
    args = add_argument()
    print("Sparsity:", args.sparsity)
    # device = "cpu"
    device = "cuda:0"
    batch_size = 256 
    nbatchs = 10 
    nsamples = batch_size * nbatchs 

    ########################## first  stage #################################
    #########################################################################
    import pickle
    teacher = args.teacher
    ori_outputs = {}
    ori_inputs = {}
    attention_mask = {}
    def create_hook(name):
        def hook(module: torch.nn.Linear, input, output):
            if name in ori_inputs:
                ori_inputs[name] = torch.cat((ori_inputs[name], input[0].cpu()), dim=0)
            else:
                ori_inputs[name] = input[0].cpu()

            if name != "bert.encoder.layer.0":
                if len(output.shape) == 2:
                    output = output.cpu().view(batch_size, -1, output.shape[-1])
                if name in ori_outputs:
                    ori_outputs[name] = torch.cat((ori_outputs[name], output.cpu()), dim=0)
                else:
                    ori_outputs[name] = output.cpu()
            else:
                # if name in attention_mask:
                #     attention_mask[name] =  torch.cat((attention_mask[name], input[1].cpu()), dim=0) 
                # else:
                attention_mask[name] = input[1].cpu()
        return hook

    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if teacher:
        with open(teacher, 'rb') as handle:
            updated_weights = pickle.load(handle)
        restore_model_weights(model, updated_weights)
    
    handlers={}
    for name, layer in model.named_modules():
        print(name)
        if type(layer) == torch.nn.Linear or name == "bert.encoder.layer.0":
            handlers[name] = layer.register_forward_hook(create_hook(name))

    train_dataset, dev_dataset, train_loader, dev_loader, test_loader = load_data(tokenizer=tokenizer, batch_size=batch_size)
    # acc, loss = evaluate(model, dev_loader, device)
    # print("Accuracy:", acc, "Loss:", loss)

    for idx, batch in enumerate(train_loader):
        print(idx)
        if idx > nbatchs:
            break
        with torch.no_grad():
            model_inputs = batch[0]
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
            model(**model_inputs)

    for name, handler in handlers.items():
        handler.remove()
        
    ########################## second stage #################################
    #########################################################################
    torch.cuda.empty_cache()
    
    if teacher:
        with open(teacher, 'rb') as handle:
            updated_weights = pickle.load(handle)
        restore_model_weights(model, updated_weights)

    mse = nn.MSELoss()
    pruned_weight = {}
    xtxs = {}
    xtys = {}
    normalize_term = {}
    xs = {}
    data_index = {"start": 0}
    # information collector hook
    def create_hook2(name):
        def hook2(module: torch.nn.Linear, input, output):
            # x and xtx
            if module.bias is not None:
                x = torch.cat((input[0], torch.ones(*(*input[0].shape[:-1], 1), device=input[0].device)), dim=-1)
            else:
                x = input[0]
            
            if name not in xs:
                xs[name] = x.cpu()
            else:
                xs[name] = torch.cat((xs[name], x.cpu()), dim=0)
            
            x = x.view(-1, x.shape[-1])
            y = ori_outputs[name][data_index["start"]:data_index["start"]+batch_size, :, :].to(device)
            
            xtx = torch.matmul(x.T, x).cpu()
            xty = torch.matmul(x.T, y.view(-1, y.shape[-1])).cpu()

            if name not in xtxs:
                xtxs[name] = xtx
            else:
                xtxs[name] += xtx

            if name not in xtys:
                xtys[name] = xty
            else:
                xtys[name] += xty
                
            if name not in normalize_term:
                normalize_term[name] = x.shape[0]
            else:
                normalize_term[name] += x.shape[0]

            return output
        return hook2

    # inputs of decoder layers 0
    inputs = ori_inputs["bert.encoder.layer.0"]
    attention_mask = attention_mask["bert.encoder.layer.0"].to(device)
    
    # recursive to prune transformer block
    for index in range(len(model.bert.encoder.layer)):
        print("*" * 10, "Block", index, "*" * 10)            
        module = model.bert.encoder.layer[index]
        target_layers = find_layers(module, name=f"bert.encoder.layer.{index}")
        
        # regiester info collecotor hook
        handlers={}
        for name, layer in module.named_modules():
            if type(layer) == torch.nn.Linear:
                ori_name = f"bert.encoder.layer.{index}.{name}"
                handlers[name] = layer.register_forward_hook(create_hook2(ori_name))
        module.to(device)
        
        # greedily to prune layers in transformer block 
        for n, l in target_layers.items():
            print("Pruning", n)
            # re-collect xtx and xty information for entire block
            with torch.no_grad():
                for input_index in range(0, nsamples, batch_size):
                    inp = inputs[input_index:input_index+batch_size, :, :].cuda()
                    res = module(inp, attention_mask=attention_mask)
                    del res
                    inp = inp.cpu()
                    torch.cuda.empty_cache()

                    data_index["start"] += batch_size

            # first batch_size inputs
            x = xs[n]
            if len(x.shape) == 2:
                x = x.view(nsamples, -1, x.shape[-1])
            x = x[:batch_size, :, :].cuda()
            
            # re-calculate best weight
            nterm = normalize_term[n]
            xtx = (xtxs[n]/nterm).cuda()
            xty = (xtys[n]/nterm).cuda()
            w_star = reconstruct_best_weight_xty(
                xtx, 
                xty, 
                w=bias_and_weights(l).cuda(), 
                ridge_regular=1e-4)
            del xty
            torch.cuda.empty_cache()

            # prune weight
            w_sparse = prune_unstructured(
                W=w_star.T,
                H=xtx,
                sparsity=args.sparsity,
            ).T
            del w_star
            torch.cuda.empty_cache()
            pruned_weight[n] = w_sparse.cpu().numpy()

            # calculate error 
            y_sparse = torch.matmul(x, w_sparse)
            ori_y = ori_outputs[n]
            if len(ori_y.shape) == 2:
                ori_y = ori_y.view(batch_size, -1, ori_y.shape[-1])
            error = mse(ori_y[:batch_size, :, :].cpu(), y_sparse.cpu())
            print("Mse error", error.item())

            # replace module weight with sparse data
            restore_layer_weights(l, w_sparse)
            del w_sparse
            torch.cuda.empty_cache()

            # clear old xtx and xty
            xtxs = {}
            xtys = {}
            xs = {}
            normalize_term = {}
            data_index = {"start": 0}
            torch.cuda.empty_cache()

        module_outputs = {}
        with torch.no_grad():
            for input_index in range(0, nsamples, batch_size):
                inp = inputs[input_index:input_index+batch_size, :, :].cuda()
                res = module(inp, attention_mask=attention_mask)
                
                if "outputs" not in module_outputs:
                    module_outputs["outputs"] = res[0].cpu()
                else:
                    module_outputs["outputs"] = torch.cat((module_outputs["outputs"], res[0].cpu()), dim=0)
                inp = inp.cpu()
                torch.cuda.empty_cache()

                data_index["start"] += batch_size
           
        xtxs = {}
        xtys = {}
        xs = {}
        normalize_term = {}
        data_index = {"start": 0}
        for key in handlers.keys():
            handlers[key].remove()

        module.to("cpu")
        del inputs         
        inputs = module_outputs["outputs"]
        torch.cuda.empty_cache()

    ########################################################################
    print_model_sparsity(model)
    for _,handler in handlers.items():
        handler.remove()

    ########################################################################
    model = model.cuda()
    acc, loss = evaluate(model, dev_loader, device)
    print("Acc", acc, "Loss", loss)
    
    ########################################################################
    if args.output:
        model.save_pretrained(args.output)
        tokenizer.save_pretrained(args.output)

if __name__ == '__main__':
    main()


