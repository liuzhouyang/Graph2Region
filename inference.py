import torch
import numpy as np
import time
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import kendalltau, spearmanr
from torch_geometric.data import Batch
from dataset_g2r import get_hard_test

def gen_batch_for_test(idx_g1, idx_g2, source, target, experiment = 'mcs'):
    g1s = []
    g2s = []
    geds = []
    mcs = []
    norm_geds = []
    bunke_geds = []
    norm_mcs = []
    bunke_mcs = []
    gu_mcs = []
    g1_size = []
    g2_size = []
    for i in range(len(idx_g1)):
        g1 = source[idx_g1[i]]
        g2 = target[idx_g2[i]]
        if g1.x == None:
            g1.x = torch.tensor(g1.num_nodes * [1.0]).unsqueeze(-1)
            g2.x = torch.tensor(g2.num_nodes * [1.0]).unsqueeze(-1)
        g1s.append(g1)
        g2s.append(g2)
        g1_size.append(g1.num_nodes)
        g2_size.append(g2.num_nodes)
        if experiment == 'ged':
            geds.append(source.ged[g1.i, g2.i])
            norm_geds.append(source.norm_ged[g1.i, g2.i])
        bunke_geds.append(source.bunke_ged[g1.i, g2.i])
        norm_mcs.append(source.norm_mcs[g1.i, g2.i])
        bunke_mcs.append(source.bunke_mcs[g1.i, g2.i])
        gu_mcs.append(source.gu_mcs[g1.i, g2.i])
    g1s = Batch.from_data_list(g1s)
    g2s = Batch.from_data_list(g2s)
    return g1s, g2s, geds, norm_geds, bunke_geds, norm_mcs, bunke_mcs, gu_mcs, g1_size, g2_size

def _cal_p_at_k(k, gt):
    gt_inc = np.sort(gt)
    tmp = (gt_inc <= gt_inc[k-1]).sum()
    if tmp > k:
        best_k_gt = gt.argsort()[:tmp]
    else:
        best_k_gt = gt.argsort()[:k]
    return best_k_gt

def cal_p_at_k(k, gt, r_pred):
    best_k_pred = r_pred[::-1][:k]
    best_k_gt = _cal_p_at_k(k, -gt)
    return len(set(best_k_pred).intersection(set(best_k_gt))) / k
    
def evaluator(gt, pred, mat, mat_gt, test):
    mse = mean_squared_error(gt, pred)
    mae = mean_absolute_error(gt, pred)
    kendall_list = []
    spearman_list = []
    p10_list = []
    kendall = 0
    spearman = 0
    p10 = 0
    if test:
        for pred, gt in zip(mat, mat_gt):
            tmp_pred = pred.argsort()
            r_pred = np.empty_like(tmp_pred)
            r_pred[tmp_pred] = np.arange(len(pred))
            
            tmp_gt = gt.argsort()
            r_gt = np.empty_like(tmp_gt)
            r_gt[tmp_gt] = np.arange(len(gt))
            kendall_list.append(kendalltau(r_pred, r_gt).statistic)
            spearman_list.append(spearmanr(r_pred, r_gt).statistic)
            p10_list.append(cal_p_at_k(10, gt, tmp_pred))
        kendall = np.mean(kendall_list).item()
        spearman = np.mean(spearman_list).item()
        p10 = np.mean(p10_list).item()
    return (mse, mae, kendall, spearman, p10)

@torch.no_grad()
def get_preds(model, loader, source, target, test, device, args):
    preds = []
    gt = []
    t0 = time.time()
    
    for pairs in loader:
        tasks = []
        start_time = time.time()
        pairs = pairs.t()
        idx_g1 = pairs[0]
        idx_g2 = pairs[1]
        g1, g2, geds, norm_geds, bunke_geds,\
             norm_mcs, bunke_mcs, gu_mcs, g1_size, g2_size = gen_batch_for_test(idx_g1, idx_g2, source, target, args.experiment)
        g1.to(device)
        g2.to(device)
        regions_g1, pe_g1 = model(g1.x, g1.edge_index)
        regions_g2, pe_g2 = model(g2.x, g2.edge_index)
        r_g1 = model.union(regions_g1, pe_g1, g1.batch)
        r_g2 = model.union(regions_g2, pe_g2, g2.batch)
        inters = model.intersection(r_g1, r_g2)
        diffs = r_g1 + r_g2 - 2 * inters
        r_g1 = torch.mean(r_g1, 0)
        r_g2 = torch.mean(r_g2, 0)
        #print(pred.squeeze(-1).shape)
        
        if args.task == 'ged':
            if args.norm_ged == 'norm':
                pred = model.predict_norm_ged(r_g1, r_g2).unsqueeze(-1)
                if args.score_rep:
                    score_rep = 2 * model.score_fc(diffs.permute(1,0,-1).flatten(1))/(torch.FloatTensor(g1_size).unsqueeze(-1).cuda() + torch.FloatTensor(g2_size).unsqueeze(-1).cuda())
                    pred = model.gamma_1 * pred + model.beta_1 * score_rep
                gt.append(torch.exp(-torch.FloatTensor(norm_geds)))
            elif args.norm_ged == 'bunke':
                gt.append(torch.FloatTensor(bunke_geds))
        if args.task == 'mcs':
            if args.norm_mcs == 'norm':
                pred = model.predict_norm_mcs(r_g1, r_g2).unsqueeze(-1)
                if args.score_rep:
                    score_rep = 2 * model.score_fc(inters.permute(1,0,-1).flatten(1))/(torch.FloatTensor(g1_size).unsqueeze(-1).cuda() + torch.FloatTensor(g2_size).unsqueeze(-1).cuda())
                    pred = model.gamma_2 * pred + model.beta_2 * score_rep
                gt.append(torch.FloatTensor(norm_mcs))
            elif args.norm_mcs == 'bunke':
                gt.append(torch.FloatTensor(bunke_mcs))
            elif args.norm_mcs == 'union':
                gt.append(torch.FloatTensor(gu_mcs))
            elif args.norm_mcs == 'value':
                gt.append(torch.FloatTensor(mcs)) 
        preds.append(pred.squeeze())
    runtime = time.time() - t0
    mse, mae = 0.0, 0.0
    pred = torch.cat(preds, dim = -1).detach().cpu().numpy()
    gt = torch.cat(gt, dim = -1).detach().cpu().numpy()
    mat = None
    mat_gt = None
    if test:
        mat = pred.reshape(len(source),len(target))
        mat_gt = gt.reshape(len(source),len(target))
    (mse, mae, kendall, spearman, p10) = evaluator(pred, gt, mat, mat_gt, test)
            #print('\nTask {}. MSE:{:.4f}. MAE:{:.4f}'.format(task, mse_mcs, mae_mcs))
    return mse, mae, kendall, spearman, p10, runtime

@torch.no_grad()
def hard_test(model, testset, device, args):
    testloader = get_hard_test(testset, args)
    testloader = iter(DataLoader(testloader, args.test_batch_size, shuffle = False))
    mse_test, mae_test, \
        kendall, spearman, \
        p10, runtime = get_preds(model, testloader, testset, testset, True, device, args)
    return mse_test, mae_test, kendall, spearman, p10, runtime

@torch.no_grad()
def test(model, testloader, graphs, graph_test, trainset, valset, testset, device, args):
    #print('starting testing')
    model.eval()
    testloader = iter(DataLoader(testloader, args.test_batch_size, shuffle = False))

    mse_test, mae_test, \
    kendall, spearman, \
    p10, runtime = get_preds(model, testloader, testset, trainset+valset, True, device, args)
    return mse_test, mae_test, kendall, spearman, p10, runtime

@torch.no_grad()
def validation(model, valloader, graphs, trainset, valset, device, args):
    #print('starting testing')
    model.eval()
    
    valloader = iter(DataLoader(valloader, args.batch_size, shuffle = True))
    mse_val, mae_val,_,_, _, _= get_preds(model, valloader, valset, trainset, False, device, args)
    return mse_val, mae_val
