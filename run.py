import torch
import random
import numpy as np

#---------------------------pytorch----------------------------------
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
#----------------------------others----------------------------------
import os
import time
from datetime import datetime
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
import yaml
#--------------------------lib---------------------------------------
from models import G2R
from inference import validation, test, hard_test
from dataset import get_dataset
import argparse
import time
import wandb

def str2bool(v):
    return v.lower() in ("true", "1")

torch.set_printoptions(precision=4)
gpu = 'cuda:0'
device = torch.device(gpu) if torch.cuda.is_available() \
                              else torch.device("cpu")
                              
def set_seeds_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
def get_loaders(trainset, valset, testset, args):
    val_mask = torch.stack((torch.arange(len(valset)).repeat_interleave(len(trainset)), 
                         torch.arange(len(trainset)).repeat(len(valset)))).t()
    test_mask = torch.stack((torch.arange(len(testset)).repeat_interleave(len(trainset) + len(valset)), 
                            torch.arange(len(trainset) + len(valset)).repeat(len(testset)))).t()
    return val_mask, test_mask

def log_builder(args):
    
    record_keys = ['name', 'dataset_name', 'experiment', 'task']
    comment = ".".join(["{}={}".format(k, v) \
              for k, v in vars(args).items() if k in record_keys])
    current_time = time.strftime("%Y-%m-%dT%H%M", time.localtime())
    logd = os.path.join('.', 'exp', args.experiment, f'g2r_{args.tag}_{args.dataset_name}_{args.task}_{current_time}')
    os.makedirs(logd, exist_ok=True)
    logger = SummaryWriter(log_dir = logd, comment = comment)
    config = vars(args)
    config["path"] = logd
    config_file_name = "config.yaml"
    with open(os.path.join(logd, config_file_name), "w") as file:
        file.write(yaml.dump(config))

    return logger, logd

def log_train(logger, task, loss, num_batches):
    logger.add_scalar(f"{task}_loss/train", loss.item(), num_batches)
    
def gen_batch(idx_g1, idx_g2, dataset, experiment = 'mcs'):
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
        g1 = dataset[idx_g1[i]]
        g2 = dataset[idx_g2[i]]
        if g1.x == None:
            g1.x = torch.tensor(g1.num_nodes * [1.0]).unsqueeze(-1)
            g2.x = torch.tensor(g2.num_nodes * [1.0]).unsqueeze(-1)
        g1s.append(g1)
        g2s.append(g2)
        g1_size.append(g1.num_nodes)
        g2_size.append(g2.num_nodes)
        if experiment == 'ged':
            geds.append(dataset.ged[g1.i, g2.i])
            norm_geds.append(dataset.norm_ged[g1.i, g2.i])
        bunke_geds.append(dataset.bunke_ged[g1.i, g2.i])
        norm_mcs.append(dataset.norm_mcs[g1.i, g2.i])
        bunke_mcs.append(dataset.bunke_mcs[g1.i, g2.i])
        gu_mcs.append(dataset.gu_mcs[g1.i, g2.i])
    g1s = Batch.from_data_list(g1s)
    g2s = Batch.from_data_list(g2s)
    return g1s, g2s, geds, norm_geds, bunke_geds, norm_mcs, bunke_mcs, gu_mcs, g1_size, g2_size
     
def train(model, optimizer, graphs, trainset, logger, run, epoch, args):
    model.train()
    trainloader = torch.randint(len(trainset),(args.batch_size*args.iterations,2))
    trainloader = iter(DataLoader(trainloader, args.batch_size, shuffle = True))
    total_loss = 0.0
    for i in range(args.iterations):
        pairs = next(trainloader)
        start_time = time.time()
        optimizer.zero_grad()
        pairs = pairs.t()
        idx_g1 = pairs[0]
        idx_g2 = pairs[1]
        g1, g2, geds, norm_geds, bunke_geds,\
             norm_mcs, bunke_mcs, gu_mcs, g1_size, g2_size = gen_batch(idx_g1, idx_g2, trainset, args.experiment)
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
        criterion = F.mse_loss
        if args.task == 'ged':
            if args.norm_ged == 'norm':
                pred = model.predict_norm_ged(r_g1, r_g2).unsqueeze(-1)
                if args.score_rep:
                    score_rep = 2 * model.score_fc(diffs.permute(1,0,-1).flatten(1))/(torch.FloatTensor(g1_size).unsqueeze(-1).cuda() + torch.FloatTensor(g2_size).unsqueeze(-1).cuda())
                    pred = model.gamma_1 * pred + model.beta_1 * score_rep
                else:
                    pred = pred
                loss = criterion(pred, torch.exp(-torch.FloatTensor(norm_geds)).to(device).unsqueeze(-1))
            elif args.norm_ged == 'bunke':
                loss = criterion(pred.unsqueeze(-1), torch.FloatTensor(bunke_geds).to(device).unsqueeze(-1))
        elif args.task == 'mcs':
            if args.norm_mcs == 'norm':
                pred = model.predict_norm_mcs(r_g1, r_g2).unsqueeze(-1)
                if args.score_rep:
                    score_rep = 2 * model.score_fc(inters.permute(1,0,-1).flatten(1))/(torch.FloatTensor(g1_size).unsqueeze(-1).cuda() + torch.FloatTensor(g2_size).unsqueeze(-1).cuda())
                    pred = model.gamma_2 * pred + model.beta_2 * score_rep
                else:
                    pred = pred
                loss = criterion(pred, torch.FloatTensor(norm_mcs).to(device).unsqueeze(-1))
            elif args.norm_mcs == 'bunke':
                # genrate bunke mcs
                loss = criterion(pred.unsqueeze(-1), torch.FloatTensor(bunke_mcs).to(device).unsqueeze(-1))
            elif args.norm_mcs == 'union':
                # genrate graph union mcs
                loss = criterion(pred.unsqueeze(-1), torch.FloatTensor(gu_mcs).to(device).unsqueeze(-1))
            elif args.norm_mcs == 'value':
                # genrate mcs 
                # TODO: provide num nodes of g1 and g2
                loss = criterion(pred.unsqueeze(-1), torch.FloatTensor(mcs).to(device).unsqueeze(-1))
        loss.backward()
        optimizer.step()
        log_train(logger, args.task, loss, i+args.eval_steps*(epoch))
        print('Iteration: {:02d}. time: {:.4f}. loss: {:.4f}. '.format(
            i+args.iterations*(epoch), time.time() - start_time, loss.item()),
            end = '                    \r'
            )
        if args.wandb:
            res_dic = {f'{args.experiment}_{args.task}_run{run}_train_loss': loss.item(),
                       f'{args.experiment}_{args.task}_run{run}_{args.task}_runtime_train': time.time() - start_time,
                       f'{args.experiment}_{args.task}_run{run}_{args.task}_iteration': i+args.iterations*(epoch)}
            wandb.log(res_dic)
        total_loss += loss
    return total_loss / args.iterations


def runG2R(args):
    print(f"executing on {device}")
    results_list = []
    hard_results_list = []
    stats = []
    if args.wandb:
        wandb.init(project='g2r', name = args.experiment + '_' + args.task +  '_g2r_' + args.dataset_name, sync_tensorboard=False)
    
    for run in range(args.runs):
        bad_epoch = 0
        print(args)
        logger, logd = log_builder(args)
        set_seeds_all(args.seed[run])
        num_node_labels, (graphs, graphs_test), (trainset, valset, testset) = get_dataset(args)
        valloader, testloader = get_loaders(trainset, valset, testset, args)
        if num_node_labels == 0:
            args.input_dim = 1
        else:
            args.input_dim = num_node_labels
        model = G2R(args)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
        mse_val = float('inf')
        mae_val = float('inf')
        best_epoch = -1
        kendall = -float('inf')
        spearman = -float('inf')
        print(f'running repetition {run}')
        t0 = time.time()
        for epoch in range(args.epochs):
            loss = train(model, optimizer, graphs, trainset, logger, run, epoch, args)
            logger.add_scalar(f'{args.experiment}_{args.task}_total_loss', loss.item(), epoch)
            if (epoch + 1) % args.eval_steps == 0 and (epoch + 1) >= args.eval_interval:
                tmp_mse_val, tmp_mae_val = validation(model, valloader, graphs, trainset, valset, device, args)
                if tmp_mse_val < mse_val:
                    mse_val = tmp_mse_val
                    mae_val = tmp_mae_val
                    best_epoch = epoch
                    bad_epoch = 0
                    if args.save_model:
                        path = f'{logd}/{args.name}_{args.dataset_name}_epoch{epoch}_{args.experiment}_{args.task}_mse_val={mse_val:.4f}.pt'
                        torch.save(model.state_dict(), path)
                        path = f'{logd}/{args.name}_{args.dataset_name}_best.pt'
                        torch.save(model.state_dict(), path)
                else:
                    bad_epoch += 1
                    
                logger.add_scalar(f"{args.experiment}_{args.task}_mse/val", tmp_mse_val, epoch)
                logger.add_scalar(f"{args.experiment}_{args.task}_mae/val", tmp_mae_val, epoch)
                logger.add_scalar(f"{args.experiment}_{args.task}_best_mse/val", mse_val, epoch)
                logger.add_scalar(f"{args.experiment}_{args.task}_best_mae/val", mae_val, epoch)
                if args.wandb:
                    res_dic = {f'{args.experiment}_{args.task}_run{run}_epoch{epoch}_total_loss': loss.item(),
                               f'{args.experiment}_{args.task}_run{run}_{args.task}_mse/val': tmp_mse_val,
                               f'{args.experiment}_{args.task}_run{run}_{args.task}_mae/val': tmp_mae_val,
                               f'{args.experiment}_{args.task}_run{run}_{args.task}_best_mse/val': mse_val,
                               f'{args.experiment}_{args.task}_run{run}_{args.task}_best_mae/val': mae_val,
                               f'{args.experiment}_{args.task}_run{run}_{args.task}_best_epoch/val': best_epoch,
                               f'{args.experiment}_{args.task}_run{run}_{args.task}_epoch/val': epoch}
                    wandb.log(res_dic)
                print('\nEpoch: {:02d}. Loss: {:.4f}.\n'
                      'Mse_val: {:.4f}. Best_mse_val: {:.4f}. \n'
                      'Mae_val: {:.4f}. Best_mae_val: {:.4f}. Best epoch: {:02d}\n'.format(
                        epoch, loss.item(), tmp_mse_val, mse_val, tmp_mae_val, mae_val, best_epoch), end='               \r')
            if bad_epoch >= args.patience:
                break
            if (time.time() - t0) // 3600 >= args.timeout:
                print('\nTimeout!\n')
                break
        path = f'{logd}/{args.name}_{args.dataset_name}_best.pt'
        del model
        model = G2R(args).cuda()
        model.load_state_dict(torch.load(path))
        mse_test, mae_test, kendall, spearman,\
             p10, runtime = test(model, testloader, graphs, graphs_test, trainset, valset, testset, device, args)
        logger.add_scalar(f"{args.experiment}_{args.task}_inference_time", runtime)
        logger.add_scalar(f"{args.experiment}_{args.task}_mse/test", mse_test)
        logger.add_scalar(f"{args.experiment}_{args.task}_mae/test", mae_test)
        logger.add_scalar(f"{args.experiment}_{args.task}_kendall/test", kendall)
        logger.add_scalar(f"{args.experiment}_{args.task}_spearman/test", spearman)
        logger.add_scalar(f"{args.experiment}_{args.task}_p@10/test", p10)
        res_dic = {f'{args.experiment}_{args.task}_run{run}_{args.task}_mse/test': mse_test,
                       f'{args.experiment}_{args.task}_run{run}_{args.task}_mae/test': mae_test,
                       f'{args.experiment}_{args.task}_run{run}_{args.task}_kendall/test': kendall,
                       f'{args.experiment}_{args.task}_run{run}_{args.task}_spearman/test': spearman,
                       f'{args.experiment}_{args.task}_run{run}_{args.task}_p@10/test': p10,
                       f'{args.experiment}_{args.task}_run{run}_{args.task}_inference/test': runtime}
            
        if args.experiment == 'mcs':
            mse_hard_test, mae_hard_test, hard_kendall, hard_spearman,\
             hard_p10, hard_runtime = hard_test(model, testset, device, args)
            logger.add_scalar(f"{args.experiment}_{args.task}_hard_inference_time", hard_runtime)
            logger.add_scalar(f"{args.experiment}_{args.task}_hard_mse/test", mse_hard_test)
            logger.add_scalar(f"{args.experiment}_{args.task}_hard_mae/test", mae_hard_test)
            logger.add_scalar(f"{args.experiment}_{args.task}_hard_kendall/test", hard_kendall)
            logger.add_scalar(f"{args.experiment}_{args.task}_hard_spearman/test", hard_spearman)
            logger.add_scalar(f"{args.experiment}_{args.task}_hard_p@10/test", hard_p10)
            tmp_dic = {f'{args.experiment}_{args.task}_run{run}_{args.task}_hard_mse/test': mse_hard_test,
                       f'{args.experiment}_{args.task}_run{run}_{args.task}_hard_mae/test': mae_hard_test,
                       f'{args.experiment}_{args.task}_run{run}_{args.task}_hard_kendall/test': hard_kendall,
                       f'{args.experiment}_{args.task}_run{run}_{args.task}_hard_spearman/test': hard_spearman,
                       f'{args.experiment}_{args.task}_run{run}_{args.task}_hard_p@10/test': hard_p10,
                       f'{args.experiment}_{args.task}_run{run}_{args.task}_hard_inference/test': hard_runtime}
            res_dic.update(tmp_dic)
        if args.wandb:
            wandb.log(res_dic)
        print('\nRun: {:02d}. Runtime: {:.4f}. \n'
              'Best_mse_val: {:.4f}. Best_mae_val: {:.4f}.\n'
              'Mse_test: {:.4f}. Mae_test: {:.4f}. \n'
              'Kendall: {:.4f}. Spearman: {:.4f}. P@10: {:.4f}. \n'
              'Inference time: {:.4f}.\n'.format(
                run, time.time() - t0, mse_val, mae_val, mse_test, mae_test, kendall, spearman, p10, runtime
            ),end = '                    \r')
        logger.add_scalar(f"memory_allocated", torch.cuda.memory_allocated())
        logger.add_scalar(f"memory_cached", torch.cuda.memory_reserved())
        torch.cuda.empty_cache()
        result = [mse_val, mae_val, mse_test, mae_test, kendall, spearman, p10]
        path = f'{logd}/result.pt'
        torch.save(result, path)
        results_list.append(result)
        if args.experiment == 'mcs':
            result = [mse_hard_test, mae_hard_test, hard_kendall, hard_spearman, hard_p10]
            hard_results_list.append(result)
        if args.runs == run + 1:
            val_mse_mean, val_mae_mean, test_mse_mean, test_mae_mean, \
                kendall_mean, spearman_mean, p10_mean = np.mean(results_list, axis=0)
            var = np.var(results_list, axis=0)
            test_mse_std = np.sqrt(var[2])
            test_mae_std = np.sqrt(var[3])
            kendall_std = np.sqrt(var[4])
            spearman_std = np.sqrt(var[5])
            p10_std = np.sqrt(var[6])
            final_result = {f'{args.experiment}_{args.task}_test_mse_mean': test_mse_mean, 
                            f'{args.experiment}_{args.task}_test_mse_std': test_mse_std,
                            f'{args.experiment}_{args.task}_test_mae_mean': test_mae_mean,
                            f'{args.experiment}_{args.task}_test_mae_std': test_mae_std, 
                            f'{args.experiment}_{args.task}_val_mse_mean': val_mse_mean, 
                            f'{args.experiment}_{args.task}_val_mae_mean': val_mae_mean, 
                            f'{args.experiment}_{args.task}_kendall_mean': kendall_mean,
                            f'{args.experiment}_{args.task}_kendall_std': kendall_std,
                            f'{args.experiment}_{args.task}_spearman_mean':spearman_mean,
                            f'{args.experiment}_{args.task}_spearman_std': spearman_std,
                            f'{args.experiment}_{args.task}_p@10_mean':p10_mean,
                            f'{args.experiment}_{args.task}_p@10_std':p10_std}
            if args.experiment == 'mcs':
                hard_test_mse_mean, hard_test_mae_mean, \
                hard_kendall_mean, hard_spearman_mean, hard_p10_mean = np.mean(hard_results_list, axis=0)
                var = np.var(hard_results_list, axis=0)
                hard_test_mse_std = np.sqrt(var[0])
                hard_test_mae_std = np.sqrt(var[1])
                hard_kendall_std = np.sqrt(var[2])
                hard_spearman_std = np.sqrt(var[3])
                hard_p10_std = np.sqrt(var[4])
                tmp_result ={f'{args.experiment}_{args.task}_hard_test_mse_mean': hard_test_mse_mean, 
                            f'{args.experiment}_{args.task}_hard_test_mse_std': hard_test_mse_std,
                            f'{args.experiment}_{args.task}_hard_test_mae_mean': hard_test_mae_mean,
                            f'{args.experiment}_{args.task}_hard_test_mae_std': hard_test_mae_std, 
                            f'{args.experiment}_{args.task}_hard_kendall_mean': hard_kendall_mean,
                            f'{args.experiment}_{args.task}_hard_kendall_std': hard_kendall_std,
                            f'{args.experiment}_{args.task}_hard_spearman_mean':hard_spearman_mean,
                            f'{args.experiment}_{args.task}_hard_spearman_std': hard_spearman_std,
                            f'{args.experiment}_{args.task}_hard_p@10_mean':hard_p10_mean,
                            f'{args.experiment}_{args.task}_hard_p@10_std':hard_p10_std}
                final_result.update(tmp_result)
            print(final_result)
            if args.wandb:
                wandb.log(final_result)
            filename = f'{logd}/results'
            with open(filename, 'a') as writefile:

                for key, value in final_result.items():
                    writefile.write(key + ' ' + str(value) +'\n')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='g2r')
   
    parser.add_argument('--dataset_name', type = str, default = 'aids',
                    help = 'name of the dataset.')
    parser.add_argument('--val_pct', type=float, default=0.2)
    parser.add_argument('--test_pct', type=float, default=0.2)
    parser.add_argument('--tag', type = str, default = None)
    parser.add_argument('--experiment', type = str, default = 'ged')
    parser.add_argument('--task', type = str, default = 'ged')
    parser.add_argument('--norm_ged', type = str, default = 'norm')
    parser.add_argument('--norm_mcs', type=str, default='norm')
    parser.add_argument('--name', type = str, default = 'g2r')
    parser.add_argument('--seed', type = int, default = [379,740,604,102,420,847,376,439,490,97],
                        help = 'Random seed number.')
    parser.add_argument('--iterations', type = int, default = 100)                  
    parser.add_argument('--batch_size', type = int, default = 128)
    parser.add_argument('--test_batch_size', type = int, default = 2048)
    parser.add_argument('--epochs', type = int, default = 5000)
    parser.add_argument('--patience', type = int, default = 50)
    parser.add_argument('--save_train', type = str2bool, default = True)
    parser.add_argument('--weight_decay', type = float, default = 0.0)
    parser.add_argument('--lr', type = float, default = 0.001,
                    help = 'Learning rate.')
    parser.add_argument('--runs', type=int, default=5, help='the number of repetition of the experiment to run')
    parser.add_argument('--eval_steps', type=int, default=20)
    parser.add_argument('--eval_interval', type=int, default=50)
    parser.add_argument('--save_model', type=str2bool, default=True)

    parser.add_argument('--wandb', type = str2bool, default = False)
    
    parser.add_argument('--num_layers', type=int, default= 8)
    parser.add_argument('--hidden_dim', type=int, default= 64)
    parser.add_argument('--layer_type', type=str, default='GIN')
    parser.add_argument('--skip_connection', type=str, default='identity')
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--output_dim', type=int, default=32)
    parser.add_argument('--alpha_type', type=str, default='learnable')
    parser.add_argument('--timeout', type=int, default=8)
    
    parser.add_argument('--num_perms', type=int, default=5)
    parser.add_argument('--length_pe', type=int, default=3)
    parser.add_argument('--max_num_nodes', type=int, default=90)
    parser.add_argument('--norm', type=str, default='layer')
    parser.add_argument('--act', type=str, default='ReLU')
    parser.add_argument('--score_rep', type=str2bool, default=True)
    parser.add_argument('--num_tasks', type=int, default=1)
    args = parser.parse_args()
    runG2R(args)
