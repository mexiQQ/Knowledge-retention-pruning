import torch
import torch.nn as nn
import numpy as np 

def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res
    
def invert(H: torch.Tensor, ridge_regular=1e-4):
    try:
        ridge = ridge_regular * torch.mean(torch.diag(H)) * torch.eye(H.shape[0], device=H.device)
        H.add_(ridge)
        del ridge
        Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
    except RuntimeError:
        return invert(H=H, ridge_regular=ridge_regular * 10)
    return Hinv

def reconstruct_best_weight(xtx: torch.Tensor, x:torch.Tensor, y:torch.Tensor, w:torch.Tensor, ridge_regular=1e-4):
    mse = nn.MSELoss()
    # for ridge_regular in [1e-4, 1e-3, 1e-2, 1e-1, 1]:
    for ridge_regular in [ridge_regular]:
        w_star = torch.matmul(torch.matmul(invert(xtx, ridge_regular=ridge_regular), x.T), y)
        error = mse(w, w_star)
        if error.item() < 1:
            # print(error.item())
            break
    return w_star

def reconstruct_best_weight_xty(xtx: torch.Tensor, xty:torch.Tensor, w:torch.Tensor, ridge_regular=1e-4):
    mse = nn.MSELoss()
    for ridge_regular in [1e-4, 1e-3, 1e-2, 1e-1, 1]:
    # for ridge_regular in [ridge_regular]:
        invert_h = invert(xtx, ridge_regular=ridge_regular)
        w_star = torch.matmul(invert_h, xty)
        del invert_h
        error = mse(w, w_star)
        if error.item() < 1:
            print("Reconstruct weight error", error.item())
            del error
            break
    return w_star
     
def bias_and_weights(module: torch.nn.Linear):
    weights = module.weight.detach().T
    if module.bias is not None:
        weights = torch.cat((weights, module.bias.detach().view(-1, weights.shape[-1])))
    return weights

def restore_layer_weights(layer, pruned_weights):
    if type(layer) == torch.nn.Linear:
        if layer.bias is not None:
            layer.weight.data = pruned_weights[:-1, :].reshape(layer.weight.data.shape[::-1]).float().T
            layer.bias.data = pruned_weights[-1, :].float()
        else:
            layer.weight.data = pruned_weights.reshape(layer.weight.data.shape[::-1]).float().T
    else:
        assert False, "Not support"
                    
def prepare_iter(i1, parallel, W, device):
    i2 = min(i1 + parallel, W.shape[0])
    count = i2 - i1
    w = W[i1:i2, :]
    mask = torch.zeros_like(w, device=device).bool()
    range_count = torch.arange(count, device=device)
    return i2, w, mask, range_count

def prepare_sparse(w, mask, Hinv):
    start = int(torch.min(torch.sum((w == 0).float(), 1)).item()) + 1
    for i in range(w.shape[0]):
        tmp = w[i] == 0
        H1 = Hinv[i]
        H1[tmp, :] = 0
        H1[:, tmp] = 0
        H1[tmp, tmp] = 1
        Hinv[i] = invert(H1)
        mask[i, torch.nonzero(tmp, as_tuple=True)[0][:(start - 1)]] = True
        del tmp, H1
    torch.cuda.empty_cache()
    return start

def prune_unstructured(W: torch.Tensor, H: torch.Tensor, parallel=128, sparsity=0.5):
    rows, columns = W.shape
    pruned_W = W.clone()
    H = H.float()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0

    if H.shape[0] > 5000:
        parallel = 8

    for i1 in range(0, rows, parallel):
        i2, w, mask, range_count = prepare_iter(i1, parallel, W, device=W.device)
        Hinv = H.unsqueeze(0).repeat((i2 - i1, 1, 1))
        start = prepare_sparse(w, mask, Hinv)
            
        for _ in range(start, int(columns * sparsity)):
            diag = torch.diagonal(Hinv, dim1=1, dim2=2)
            scores = (w ** 2) / diag
            scores[mask] = float('inf')
            j = torch.argmin(scores, 1)
            
            row = Hinv[range_count, j, :]
            d = diag[range_count, j]
            w.sub_(row * (w[range_count, j] / d).unsqueeze(1))
            mask[range_count, j] = True
            w[mask] = 0
            
            row.div_(torch.sqrt(d).unsqueeze(1))
            Hinv.sub_(torch.bmm(row.unsqueeze(2), row.unsqueeze(1)))

            del diag, scores, j, row, d
            torch.cuda.empty_cache()

        pruned_W[i1:i2, :] = w
        del i2, w, mask, range_count, Hinv
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    return pruned_W

def prune_nmstructured(W: torch.Tensor, H: torch.Tensor, parallel=128, sparsity=0.5, m=64):
    n = int(m * sparsity)
    rows, columns = W.shape
    pruned_W = W.clone()
    H = H.float()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0

    if H.shape[0] > 5000:
        parallel = 8

    for i1 in range(0, rows, parallel):
        i2, w, mask, range_count = prepare_iter(i1, parallel, W, device=W.device)
        Hinv = H.unsqueeze(0).repeat((i2 - i1, 1, 1))
        start = prepare_sparse(w, mask, Hinv)
            
        banks = torch.zeros((i2-i1, columns // m, 1), device=W.device)
        for _ in range(start, int(columns * sparsity)):
            diag = torch.diagonal(Hinv, dim1=1, dim2=2)
            scores = (w ** 2) / diag
            tmp = (banks >= n).repeat((1, 1, m)).flatten(1)
            mask[:, :-1] = mask[:, :-1] | tmp 
            scores[mask] = float('inf')
            j = torch.argmin(scores, 1)
            
            row = Hinv[range_count, j, :]
            d = diag[range_count, j]
            w.sub_(row * (w[range_count, j] / d).unsqueeze(1))
            mask[range_count, j] = True
            w[mask] = 0
            
            row.div_(torch.sqrt(d).unsqueeze(1))
            Hinv.sub_(torch.bmm(row.unsqueeze(2), row.unsqueeze(1)))
            # import pdb
            # pdb.set_trace()
            
            idx = torch.div(j, m, rounding_mode='floor')
            mask2 = idx < banks.size(1)
            # Apply the mask to idx, range_count and banks
            idx = idx[mask2]
            range_count2 = range_count[mask2]
            banks[range_count2, idx, 0] += 1

            del diag, scores, j, row, d, tmp
            torch.cuda.empty_cache()

        pruned_W[i1:i2, :] = w
        del i2, w, mask, range_count, Hinv
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    return pruned_W

