#---------------------------pytorch----------------------------------
import torch
from torch.nn import Module, Parameter, Linear, Embedding, Sequential, Dropout, MultiheadAttention
from torch_geometric.utils import to_dense_batch, scatter, add_self_loops
import torch.nn.functional as F
import random
from torch_geometric.nn.glob import global_add_pool, global_mean_pool

from layers import GNN, GSinkPropagation, norm_act_drop

class G2R(Module):
    def __init__(self, args):
        super(G2R, self).__init__()
        self.length_pe = args.length_pe
        self.num_perms = args.num_perms
        self.max_num_nodes = args.max_num_nodes
        self.gnn = GNN(args)
        self.perm = Embedding(self.max_num_nodes, self.num_perms)
        self.pe_prop = GSinkPropagation('max')
        self.fc = Sequential(Linear(args.hidden_dim, int(args.hidden_dim / 2)),
                              norm_act_drop(int(args.hidden_dim / 2), args.norm, args.act, 0.0),
                              Linear(int(args.hidden_dim / 2), args.output_dim))
        self.positional = Sequential(Linear(self.length_pe * self.num_perms, int(self.length_pe * self.num_perms * 2)),
                                     norm_act_drop(int(self.length_pe * self.num_perms *2), args.norm, args.act, 0.0),
                                     Linear(int(self.length_pe * self.num_perms *2), args.output_dim))
        self.lin = Linear(args.output_dim, args.output_dim)
        if args.alpha_type == 'learnable' or args.alpha_type == 'sigmoid':
            self.alpha = Parameter(torch.ones(1))
        elif args.alpha_type == 'linear':
            self.alpha = Linear(1, 1)
        self.alpha_type = args.alpha_type
        self.weights = Parameter(torch.zeros([args.num_tasks]))
        
        self.dataset_name = args.dataset_name
        self.beta_1 = Parameter(torch.ones(1))
        self.beta_2 = Parameter(torch.ones(1))
        self.gamma_1 = Parameter(torch.ones(1))
        self.gamma_2 = Parameter(torch.ones(1))
        if args.score_rep:
            self.beta = Parameter(torch.ones(1))
            self.num_channels = args.output_dim * args.num_layers
            self.score_fc =  Sequential(Linear(self.num_channels, self.num_channels // 4),
                              norm_act_drop(self.num_channels // 4, args.norm, args.act, args.dropout),
                              Linear(self.num_channels // 4, self.num_channels // 16),
                              norm_act_drop(self.num_channels // 16, args.norm, args.act, args.dropout),
                              Linear(self.num_channels // 16, 1))

    def forward(self, x, edge_index):
        # 1. calculate k-hop embeddings for each node
        xs = self.gnn(x, edge_index)
        regions = self.fc(xs)
        # 2. generate positional embs for each node
        pe = self.origin_gen(x, edge_index)

        return regions, pe

    def node2region(self, x, edge_index):
        xs = self.gnn(x, edge_index)
        regions = self.fc(xs)
        return regions
    
    def origin_gen(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index)

        # 1. assign each node a random number in each permutation
        # walking length, number of nodes, number of permutations
        coors = torch.zeros((self.length_pe, x.size(0), self.num_perms)).to(x.device) # initialize container
        # assign each node a random index
        idx = torch.tensor(random.choices(range(self.max_num_nodes), k = x.size(0)), requires_grad=False)
        # generate k permutation according to the random index
        coors[0] = self.perm(idx.cuda()).detach()
        # 2. propagate the max number
        for i in range(self.length_pe-1):
            coors[i+1] = self.pe_prop(coors[i], edge_index)

        # number of nodes, number of permutations, walking length
        # trans_coors = torch.transpose(torch.transpose(coors, 1, 0), -1, -2)
        trans_coors = coors.permute(1, -1, 0)
        num_nodes, num_perm, len_pe = trans_coors.shape
        # number of nodes, number of permutations * walking length
        trans_coors = trans_coors.reshape(num_nodes, num_perm * len_pe)
        pe = self.positional(trans_coors)
        return pe
    
    def union(self, regions, offsets, batch):
        # choose smallest offsets and biggest x at each dim
        # smallest convex of regions
        # [n, num_nodes_per_batch, hidden_dim]
        r_dim = -1 if regions.dim() == 1 else -2
        o_dim = -1 if offsets.dim() == 1 else -2
        # size = int(batch.max().item() + 1)
        # calculate batch-wise offset and region by taking the channal-wise maximum
        # across the region dimension and the minimum .. the offset dimension
        # dim_size will be automatically calculated in scatter
        regions += offsets
        origin = scatter(offsets, batch, dim=o_dim, dim_size=None, reduce='min')
        #region = scatter(regions, batch, dim=r_dim, dim_size=None, reduce='max')
        
        region = global_add_pool(regions, batch)# one linux on this run
        r = region - origin
        # [n, num of graphs, hidden_dim]
        r = F.normalize(r, dim=-1)# for imdb or nan
        return self.lin(r)

    def intersection(self, region1, region2):
        return torch.min(region1, region2)
    
    # nMCS = mcs(g1, g2) / (|g1| + |g2|)/2 -> vol of inter / ((vol of g1 + vol of g2) / 2.0)
    def predict_norm_mcs(self, region1, region2):
        inter = self.intersection(region1, region2)
        vol_inter = self.cal_volume(inter) - 1.0
        vol_r1 = self.cal_volume(region1)
        vol_r2 = self.cal_volume(region2)
        return vol_inter / ((vol_r1 + vol_r2)/2.0)
        #return self.beta * vol_inter / ((vol_r1 + vol_r2)/2.0)
 
    # nGED = exp(-alpha * bunkeGED(g1, g2) / (|g1| + |g2|) / 2) -> exp(-bunkeGED(g1, g2) / ((vol of g1 + vol of g2) / 2.0))
    # bunkeGED(g1, g2) = |g1| + |g2| - 2 * mcs(g1, g2) -> vol of g1 + vol of g2 - 2 * vol of inter
    def predict_norm_ged(self, region1, region2):
        # based on bunke ged
        inter = self.intersection(region1, region2)
        vol_inter = self.cal_volume(inter) - 1.0
        vol_r1 = self.cal_volume(region1)
        vol_r2 = self.cal_volume(region2)
        bunke_ged = vol_r1 + vol_r2 - 2.0 * vol_inter
        if self.alpha_type == 'learnable':
            return torch.exp(-self.alpha * bunke_ged / (vol_r1 + vol_r2 / 2.0))
        elif self.alpha_type == 'none':
            return torch.exp(-bunke_ged / (vol_r1 + vol_r2 / 2.0))
        elif self.alpha_type == 'linear':
            return torch.exp(-self.alpha(bunke_ged.unsqueeze(-1)).squeeze(1) / (vol_r1 + vol_r2 / 2.0))
        elif self.alpha_type == 'sigmoid':
            return torch.exp(-torch.sigmoid(self.alpha) * bunke_ged / (vol_r1 + vol_r2 / 2.0))
    
    # nbunkeGED = exp(-(|g1| + |g2| - 2 * mcs(g1, g2)) / (|g1| + |g2|) / 2) -> 
    # exp(-(vol of g1 + vol of g2 - 2 * vol of inter) / ((vol of g1 + vol of g2) / 2.0)))
    def predict_bunke_ged(self, region1, region2):
        inter = self.intersection(region1, region2)
        vol_inter = self.cal_volume(inter) - 1.0
        vol_r1 = self.cal_volume(region1)
        vol_r2 = self.cal_volume(region2)
        bunke_ged = vol_r1 + vol_r2 - 2.0 * vol_inter
        return torch.exp(-bunke_ged / (vol_r1 + vol_r2 / 2.0))

    # bunkeMCS = mcs(g1, g2) / max(|g1|, |g2|) -> vol of inter / max(vol of g1, vol of g2)
    def predict_bunke_mcs(self, region1, region2):
        inter = self.intersection(region1, region2)
        vol_inter = self.cal_volume(inter) - 1.0
        vol_r1 = self.cal_volume(region1)
        vol_r2 = self.cal_volume(region2)
        return vol_inter / torch.max(vol_r1, vol_r2)
    
    # graphUMCS = mcs(g1, g2) / (|g1| + |g2| - mcs(g1, g2)) -> vol of inter / (vol of g1 + vol of g2 - vol of inter)
    def predict_graph_union_mcs(self, region1, region2):
        inter = self.intersection(region1, region2)
        vol_inter = self.cal_volume(inter) - 1.0
        vol_r1 = self.cal_volume(region1)
        vol_r2 = self.cal_volume(region2)
        return vol_inter / (vol_r1 + vol_r2 - vol_inter)

    def cal_volume(self, regions):
        # could be zero when one dimension appears to be zero
        # regions = torch.clamp(regions, min = 1e-8)
        vol = torch.prod(torch.exp(regions).sqrt(), -1)
        return torch.max(torch.zeros_like(vol), vol)
