import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

class TransE(nn.Module):
    def __init__(self, entity_count, relation_count, embedding_dim, margin=1.0, norm=1):
        super(TransE, self).__init__()
        self.entity_embeddings = nn.Embedding(entity_count, embedding_dim)
        self.relation_embeddings = nn.Embedding(relation_count, embedding_dim)
        nn.init.uniform_(self.entity_embeddings.weight, -6/np.sqrt(embedding_dim), 6/np.sqrt(embedding_dim))
        nn.init.uniform_(self.relation_embeddings.weight, -6/np.sqrt(embedding_dim), 6/np.sqrt(embedding_dim))
        self.relation_embeddings.weight.data = nn.functional.normalize(
            self.relation_embeddings.weight.data, p=2, dim=1)
        self.margin = margin
        self.norm = norm
    
    def forward(self, heads, relations, tails):
        h = self.entity_embeddings(heads)
        r = self.relation_embeddings(relations)
        t = self.entity_embeddings(tails)
        h = nn.functional.normalize(h, p=2, dim=1)
        t = nn.functional.normalize(t, p=2, dim=1)
        score = torch.norm(h + r - t, p=self.norm, dim=1)
        return score

class RGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_relations, num_bases=None):
        super(RGCNLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_relations = num_relations
        
        if num_bases is None or num_bases > num_relations:
            self.weight = nn.Parameter(torch.FloatTensor(
                num_relations, in_dim, out_dim))
            nn.init.xavier_uniform_(self.weight)
        else:
            self.num_bases = num_bases
            self.bases = nn.Parameter(torch.FloatTensor(num_bases, in_dim, out_dim))
            self.weights = nn.Parameter(torch.FloatTensor(num_relations, num_bases))
            nn.init.xavier_uniform_(self.bases)
            nn.init.xavier_uniform_(self.weights)
        
        self.bias = nn.Parameter(torch.FloatTensor(out_dim))
        nn.init.zeros_(self.bias)
    
    def forward(self, node_features, adj_list):
        outputs = []
        for rel in range(self.num_relations):
            adj = adj_list[rel]
            
            if hasattr(self, 'bases'):
                weight = torch.matmul(self.weights[rel:rel+1], 
                                     self.bases.view(self.num_bases, -1)).view(1, self.in_dim, self.out_dim)
            else:
                weight = self.weight[rel:rel+1]
            
            if adj.is_sparse:
                message = torch.sparse.mm(adj, node_features)
            else:
                message = torch.matmul(adj, node_features)
            
            rel_output = torch.matmul(message, weight.squeeze(0))
            outputs.append(rel_output)
        
        output = torch.sum(torch.stack(outputs, dim=0), dim=0)
        output = output + self.bias
        
        return torch.relu(output)

class RGCN(nn.Module):
    def __init__(self, num_nodes, hidden_dim, output_dim, num_relations, num_bases=None, num_layers=2):
        super(RGCN, self).__init__()
        self.embedding = nn.Embedding(num_nodes, hidden_dim)
        nn.init.xavier_uniform_(self.embedding.weight)
        
        self.layers = nn.ModuleList()
        self.layers.append(RGCNLayer(hidden_dim, hidden_dim, num_relations, num_bases))
        for _ in range(num_layers - 2):
            self.layers.append(RGCNLayer(hidden_dim, hidden_dim, num_relations, num_bases))
        self.layers.append(RGCNLayer(hidden_dim, output_dim, num_relations, num_bases))
    
    def forward(self, adj_list):
        h = self.embedding.weight
        
        for layer in self.layers:
            h = layer(h, adj_list)
        
        return h

class RGCNTransE(nn.Module):
    def __init__(self, num_nodes, num_relations, rgcn_dim, transe_dim):
        super(RGCNTransE, self).__init__()
        self.rgcn = RGCN(num_nodes, rgcn_dim, rgcn_dim, num_relations)
        self.transe = TransE(num_nodes, num_relations, transe_dim)
        self.final_dim = rgcn_dim + transe_dim
    
    def get_combined_embeddings(self, adj_list):
        rgcn_embeddings = self.rgcn(adj_list)
        transe_embeddings = self.transe.entity_embeddings.weight
        combined_embeddings = torch.cat([rgcn_embeddings, transe_embeddings], dim=1)
        return nn.functional.normalize(combined_embeddings, p=2, dim=1)

def train_transe(triplets, entity_count, relation_count, embedding_dim, kge_epochs, batch_size=128, lr=0.001):
    model = TransE(entity_count, relation_count, embedding_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(kge_epochs):
        total_loss = 0
        np.random.shuffle(triplets)
        for i in range(0, len(triplets), batch_size):
            batch = triplets[i:i+batch_size]
            heads, relations, tails = batch[:, 0], batch[:, 1], batch[:, 2]
            neg_heads = heads.clone()
            neg_tails = tails.clone()
            for j in range(len(batch)):
                if np.random.random() < 0.5:
                    neg_heads[j] = np.random.randint(0, entity_count)
                else:
                    neg_tails[j] = np.random.randint(0, entity_count)    
            pos_score = model(heads, relations, tails)
            neg_score = model(neg_heads, relations, neg_tails)
            loss = torch.mean(torch.clamp(pos_score - neg_score + 1, min=0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()   
            total_loss += loss.item() 
        if (epoch + 1) % 10 == 0:
            print(f"TransE Epoch {epoch+1}/{epochs}, Loss: {total_loss/(len(triplets)//batch_size)}")
    
    return model

def build_adjacency_matrices(triplets, num_nodes, num_relations):
    adj_list = []
    for rel in range(num_relations):
        rel_triplets = triplets[triplets[:, 1] == rel]
        if len(rel_triplets) == 0:
            indices = torch.LongTensor([[], []])
            values = torch.FloatTensor([])
            adj = torch.sparse.FloatTensor(indices, values, (num_nodes, num_nodes))
        else:
            rows = rel_triplets[:, 0]
            cols = rel_triplets[:, 2]
            indices = torch.stack([torch.LongTensor(rows), torch.LongTensor(cols)])
            values = torch.ones(len(rel_triplets))
            adj = torch.sparse.FloatTensor(indices, values, (num_nodes, num_nodes))
        
        adj_list.append(adj)
    
    return adj_list

def train_rgcn(triplets, num_nodes, num_relations, hidden_dim, output_dim, gnn_epochs, lr=0.01):
    adj_list = build_adjacency_matrices(triplets, num_nodes, num_relations)
    
    model = RGCN(num_nodes, hidden_dim, output_dim, num_relations)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(gnn_epochs):
        node_embeddings = model(adj_list)
        
        loss = 0
        for rel in range(num_relations):
            adj = adj_list[rel]
            if adj._nnz() > 0:
                sim = torch.mm(node_embeddings, node_embeddings.t())
                if adj.is_sparse:
                    loss += torch.norm(adj.to_dense() - torch.sigmoid(sim))
                else:
                    loss += torch.norm(adj - torch.sigmoid(sim))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"RGCN Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
    
    return model, adj_list
