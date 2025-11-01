import torch
from torch_geometric.utils import to_networkx
from torch_geometric.data import InMemoryDataset, Data, download_url, \
                                 extract_tar, extract_zip, Batch

from torch_geometric.datasets import GEDDataset
import os.path as osp

import networkx as nx

import random

class MCSDataset(InMemoryDataset):
    def __init__(self, root: str, name: str, transform = None,
                 pre_transform = None, pre_filter = None):
        self.name = name
        super().__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0]
        self.data, self.slices = torch.load(path)
        path = osp.join(self.processed_dir, f'{self.name}_bunke_ged_similarity_mat.pt')
        self.bunke_ged = torch.load(path)
        path = osp.join(self.processed_dir, f'{self.name}_norm_mcs_similarity_mat.pt')
        self.norm_mcs = torch.load(path)
        path = osp.join(self.processed_dir, f'{self.name}_bunke_mcs_similarity_mat.pt')
        self.bunke_mcs = torch.load(path)
        path = osp.join(self.processed_dir, f'{self.name}_graph_union_mcs_similarity_mat.pt')
        self.gu_mcs = torch.load(path)

    @property
    def raw_file_names(self):
        return  f'{self.name}/'

    @property
    def raw_dir(self) -> str:
        name = f'raw'
        return osp.join(self.root, name)
    
    @property
    def processed_file_names(self):
        names = ['dataset', 'mcs', 'norm_mcs_similarity_mat', 'bunke_mcs_similarity_mat',
                'graph_union_mcs_similarity_mat', 'bunke_ged_similarity_mat']
        return [f'{self.name}_{name}.pt' for name in names]

    def process(self):
        graphs = torch.load(osp.join(self.raw_dir, '5_None_train.pt'))
        dataset = torch.load(osp.join(self.raw_dir, '5_None_graphs.pt'))
        data_list = []
        Ns = [graphs[i].number_of_nodes() for i in range(len(graphs))]
        N = torch.tensor(Ns, dtype=torch.float)
        if not osp.exists(self.processed_paths[0]):
            for i in range(len(graphs)):
                if graphs[i].number_of_nodes() > 3:
                    data = Data(x=dataset[i].x, edge_index=dataset[i].edge_index, i=i)
                    data.num_nodes = graphs[i].number_of_nodes()
                    
                    data_list.append(data)

                torch.save(self.collate(data_list), self.processed_paths[0])
        if not osp.exists(osp.join(self.raw_dir, f'{self.name}_mcs.pt')):
            mat = torch.full((len(graphs), len(graphs)), float(0))
            for i in range(len(graphs)-1):
                mat[i,i] = graphs[i].number_of_nodes()
                path = './MCS/' + self.name + '/raw/results_per_graph/' + \
                               'graph' + str(i) + '/num_nodes'
                with open(path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        j = int(line.split(' ')[0])
                        mcs = int(line.split(' ')[1])
                        mat[i, j], mat[j, i] = mcs, mcs
            mat[-1][-1] = graphs[-1].number_of_nodes()
            torch.save(mat, osp.join(self.raw_dir, f'{self.name}_mcs.pt'))
            torch.save(mat, self.processed_paths[1])
        else:
            mat = torch.load(osp.join(self.raw_dir, f'{self.name}_mcs.pt'))
        # Calculate the normalized MCSs:
        
        if not osp.exists(self.processed_paths[2]):
            norm_mat = mat / (0.5 * (N.view(-1, 1) + N.view(1, -1)))

            path = osp.join(self.processed_dir, f'{self.name}_norm_mcs_similarity_mat.pt')
            torch.save(norm_mat, self.processed_paths[2])

        # Calculate the bunke MCSs:
        if not osp.exists(self.processed_paths[3]):
            norm_mat = mat / (torch.max(N.view(-1, 1),N.view(1, -1)))
            path = osp.join(self.processed_dir, f'{self.name}_bunke_mcs_similarity_mat.pt')
            torch.save(norm_mat, self.processed_paths[3])

        # Calculate the graph union MCSs:
        if not osp.exists(self.processed_paths[4]):
            norm_mat = mat / ((N.view(-1, 1) + N.view(1, -1))-mat)

            path = osp.join(self.processed_dir, f'{self.name}_graph_union_mcs_similarity_mat.pt')
            torch.save(norm_mat, self.processed_paths[4])

        # Calculate the bunke GEDs:
        if not osp.exists(self.processed_paths[5]):
            print('Generating bunke GEDs...')
            mat = torch.full((len(graphs), len(graphs)), float(0.0))
            o = torch.full((len(graphs), len(graphs)), float(0.0))
            from tqdm import tqdm
            for i in tqdm(range(len(graphs)-1)):
                path = './MCS/' + self.name + '/raw/results_per_graph/' + \
                                   'graph' + str(i) + '/solutions'
                with open(path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        line = line.split(' ')
                        j = int(line[0])
                        s1 = []
                        s2 = []
                        for match in range(1, len(line)):
                            m =  line[match].split(':')
                            if len(m) > 1:
                                s1.append(int(m[0]))
                                s2.append(int(m[1]))
                        overlap1 = graphs[i].subgraph(s1)
                        overlap2 = graphs[j].subgraph(s2)
                        if overlap1.number_of_edges() < overlap2.number_of_edges():
                            overlap = overlap1
                        else:
                            overlap = overlap2
                        overlap = nx.convert_node_labels_to_integers(overlap)
                        nn = overlap.number_of_nodes()
                        ne = overlap.number_of_edges()
                        overlap = nn + ne # mcs with edges
                        d = graphs[i].number_of_nodes() + graphs[i].number_of_edges() +\
                                graphs[j].number_of_nodes() + graphs[j].number_of_edges() - 2 * overlap
                        mat[i, j], mat[j, i] = d, d
                        o[i,j], o[j,i] = overlap, overlap
            o[-1][-1] = graphs[-1].number_of_nodes()
            torch.save(o, osp.join(self.raw_dir, f'{self.name}_mcs_with_edges.pt'))
            norm_mat = torch.exp(-(mat) / (0.5 * (N.view(-1, 1) + N.view(1, -1))))

            path = osp.join(self.processed_dir, f'{self.name}_bunke_ged_similarity_mat.pt')
            torch.save(norm_mat, self.processed_paths[5])

    def __repr__(self) -> str:
        return f'{self.name}({len(self)})'

def load_ged_dataset(args, train = True):
    if args.dataset_name == "aids":
        dataset = GEDDataset(root="./GED/AIDS700nef",
                            name="AIDS700nef", train = train)
    elif args.dataset_name == "linux":
        dataset = GEDDataset(root="./GED/LINUX",
                            name="LINUX", train = train)
    elif args.dataset_name == "imdb":
        dataset = GEDDataset(root="./GED/IMDBMulti",
                            name="IMDBMulti", train = train)
    path = osp.join(dataset.processed_dir, f'{dataset.name}_bunke_ged_with_edges_similarity_mat.pt')
    dataset.bunke_ged = torch.load(path)
    path = osp.join(dataset.processed_dir, f'{dataset.name}_norm_mcs_similarity_mat.pt')
    dataset.norm_mcs = torch.load(path)
    path = osp.join(dataset.processed_dir, f'{dataset.name}_bunke_mcs_similarity_mat.pt')
    dataset.bunke_mcs = torch.load(path)
    path = osp.join(dataset.processed_dir, f'{dataset.name}_graph_union_mcs_similarity_mat.pt')
    dataset.gu_mcs = torch.load(path)
    if dataset[0].x == None:
        num_node_labels = 0
    else:
        num_node_labels = dataset[0].x.size(-1)
    graphs = []
    for i, data in enumerate(dataset):
        if not type(data) == nx.Graph:
            if num_node_labels == 0:
                graph = to_networkx(data).to_undirected()
            else:
                graph = to_networkx(data, node_attrs = ['x']).to_undirected()
        graphs.append(graph)
    if train:
        train_len = int(len(dataset) * (1.0 - args.val_pct))
        return num_node_labels, graphs, dataset[:train_len], dataset[train_len:]
    else:
        return num_node_labels, graphs, dataset

def load_mcs_dataset(name, val_pct):
    dataset = MCSDataset(root=osp.join('./MCS/', name),
                             name=name)
    if dataset[0].x == None:
        num_node_labels = 0
    else:
        num_node_labels = dataset[0].x.size(-1)
    graphs = []
    for i, data in enumerate(dataset):
        if not type(data) == nx.Graph:
            if num_node_labels == 0:
                graph = to_networkx(data).to_undirected()
            else:
                graph = to_networkx(data, node_attrs = ['x']).to_undirected()
            graphs.append(graph)
    train_len = int(len(dataset) * 0.8 * (1.0 - val_pct))
    val_len = int(len(dataset) * 0.8) - train_len
    return num_node_labels, graphs[:train_len+val_len], graphs[train_len+val_len:], \
           dataset[:train_len], dataset[train_len:train_len+val_len], dataset[train_len+val_len:]

def get_dataset(args):
    if args.dataset_name in ['aids', 'linux', 'imdb'] and args.experiment == 'ged':
        num_node_labels, graphs, trainset, valset = load_ged_dataset(args, train = True)
        _, graphs_test, testset = load_ged_dataset(args, train = False)
        
    else:
        num_node_labels, graphs, graphs_test, \
              trainset, valset, testset = load_mcs_dataset(args.dataset_name, args.val_pct)
    return num_node_labels, (graphs, graphs_test), (trainset, valset, testset)

def get_loaders(trainset, valset, testset, args):
    train_mask = torch.randint(len(trainset),(args.batch_size*args.epochs,2))
    val_mask = torch.stack((torch.arange(len(valset)).repeat_interleave(len(trainset)), 
                         torch.arange(len(trainset)).repeat(len(valset)))).t()
    test_mask = torch.stack((torch.arange(len(testset)).repeat_interleave(len(trainset) + len(valset)), 
                            torch.arange(len(trainset) + len(valset)).repeat(len(testset)))).t()
    return train_mask, val_mask, test_mask

def get_hard_test(testset, args):
    test_mask = torch.stack((torch.arange(len(testset)).repeat_interleave(len(testset)), 
                             torch.arange(len(testset)).repeat(len(testset)))).t()
    return test_mask

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

