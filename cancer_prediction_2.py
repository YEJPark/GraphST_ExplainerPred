import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from torch_geometric.nn import GNNExplainer

# TSV 파일에서 노드 및 엣지 데이터 불러오기
node_df = pd.read_csv('nodes.tsv', sep='\t')
edge_df = pd.read_csv('edges.tsv', sep='\t')

# 노드 데이터 생성
node_features = torch.tensor(node_df[['cell_id', 'gene_id', 'localization', 'cancer']].values, dtype=torch.float)

# gene names 매핑 정보
gene_names = ['Gene1', 'Gene2', 'Gene3']

# localization 매핑 정보
localization_mapping = {0: 'Nucleus', 1: 'Cytoplasm', 2: 'Golgi'}

# 엣지 데이터 생성
edge_index = torch.tensor(edge_df.values.T, dtype=torch.long)

# Graph data 생성
cell_ids = node_df['cell_id'].unique()
data_list = []
for cell_id in cell_ids:
    mask = node_df['cell_id'] == cell_id
    cell_node_indices = mask[mask].index.tolist()
    sub_edge_index = []
    for edge in edge_index.T:
        if edge[0] in cell_node_indices and edge[1] in cell_node_indices:
            sub_edge_index.append(edge)
    sub_edge_index = torch.tensor(sub_edge_index, dtype=torch.long).T
    sub_data = Data(x=node_features[cell_node_indices], edge_index=sub_edge_index)
    data_list.append(sub_data)

# GAT 모델 정의
class GATModel(torch.nn.Module):
    def __init__(self):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(4, 16, heads=4)  # 4개의 노드 특성, 16개의 출력 특성, 4개의 attention heads
        self.conv2 = GATConv(16 * 4, 2, heads=4)  # 16*4개의 입력 특성, 2개의 출력 특성, 4개의 attention heads
        self.out = torch.nn.Linear(2 * 4, 1)  # 2*4개의 입력 특성, 1개의 출력 특성

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = global_mean_pool(x, torch.zeros(x.size(0), dtype=torch.long))  # 각 cell에 대해 풀링
        x = self.out(x)
        return torch.sigmoid(x)

# 모델 인스턴스 생성
model = GATModel()

# 데이터 로더와 학습 루프
def train(model, data_list, epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data in data_list:
            optimizer.zero_grad()
            out = model(data)
            loss = F.binary_cross_entropy(out, data.x[:, -1].view(-1, 1))  # 각 cell의 cancer 여부
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(data_list):.4f}")

# 학습
train(model, data_list)

# 예측 함수 정의
def predict(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data)
        prediction = (out > 0.5).float()
    return prediction

# GNNExplainer를 사용하여 중요 노드 평가 함수 정의
def evaluate_gene_importance(model, data_list):
    explainer = GNNExplainer(model, epochs=200)
    node_feat_mask_list = []
    edge_mask_list = []
    for data in data_list:
        node_feat_mask, edge_mask = explainer.explain_graph(data)
        node_feat_mask_list.append(node_feat_mask)
        edge_mask_list.append(edge_mask)
    return node_feat_mask_list, edge_mask_list

# 예측 및 중요도 평가
for i, data in enumerate(data_list):
    prediction = predict(model, data)
    print(f"Predictions for cell {i}: ", prediction)

node_feat_mask_list, edge_mask_list = evaluate_gene_importance(model, data_list)

for i, node_feat_mask in enumerate(node_feat_mask_list):
    print(f"Node Feature Importance for cell {i}: ", node_feat_mask)

for i, edge_mask in enumerate(edge_mask_list):
    print(f"Edge Importance for cell {i}: ", edge_mask)

# gene importance 및 localization 정보 출력
for i, node_feat_mask in enumerate(node_feat_mask_list):
    gene_importance = {}
    for j in range(node_feat_mask.shape[0]):
        gene_id = int(data_list[i].x[j, 1].item())
        localization = localization_mapping[int(data_list[i].x[j, 2].item())]
        importance = node_feat_mask[j].mean().item()
        if gene_id not in gene_importance:
            gene_importance[gene_id] = {'importance': [], 'localization': localization}
        gene_importance[gene_id]['importance'].append(importance)

    for gene_id, info in gene_importance.items():
        avg_importance = np.mean(info['importance'])
        print(f"Gene {gene_names[gene_id-1]} importance for cell {i}: {avg_importance:.4f}, Localization: {info['localization']}")

# 전체 노드 feature 중요도 평가
for i, node_feat_mask in enumerate(node_feat_mask_list):
    feature_importance = node_feat_mask.mean(dim=0)
    for j, importance in enumerate(feature_importance):
        print(f"Feature {j} importance for cell {i}: {importance.item():.4f}")
