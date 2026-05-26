import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch

class FocusAwareGraphBuilder(nn.Module):
    """
    创新功能1实现：基于聚焦先验 + 滑动窗口的高效构图模块
    """
    def __init__(self, k=8, window_size=7, gamma=2.0):
        super().__init__()
        self.k = k
        self.window_size = window_size
        self.gamma = gamma
        # 固定拉普拉斯核用于计算聚焦度
        self.register_buffer('lap_kernel', torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], dtype=torch.float32))

    def get_laplacian_magnitude(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        kernel = self.lap_kernel.repeat(C, 1, 1, 1).to(x.device)
        with torch.no_grad():
            lap = F.conv2d(x, kernel, padding=1, groups=C)
            # [B, 1, H, W] 平均各通道的梯度幅值
            lap_mag = torch.mean(torch.abs(lap), dim=1, keepdim=True)
        return lap_mag

    def forward(self, features):
        """
        Args:
            features: [B, C, H, W] CNN提取的特征
        Returns:
            edge_index: [2, E] (全Batch的边索引，已处理Batch偏移)
        """
        B, C, H, W = features.shape
        device = features.device
        N = H * W
        pad = self.window_size // 2

        # 1. 计算聚焦先验图 [B, 1, H, W]
        focus_map = self.get_laplacian_magnitude(features)
        focus_map = (focus_map - focus_map.min()) / (focus_map.max() - focus_map.min() + 1e-6)

        # 2. 特征归一化
        feat_norm = F.normalize(features, p=2, dim=1)

        edge_indices_list = []
        
        # 逐个样本构图 (为了代码清晰，Batch循环处理，实际部署可用并行Unfold优化)
        for b in range(B):
            # --- 创新点：Unfold 滑动窗口 ---
            # Center: [N, 1, C]
            f_center = feat_norm[b].view(C, -1).permute(1, 0).unsqueeze(1) 
            # Neighbors: Unfold -> [C, K*K, N] -> [N, C, K*K]
            f_unfold = F.unfold(feat_norm[b].unsqueeze(0), self.window_size, padding=pad)
            f_neighbors = f_unfold.view(C, self.window_size**2, N).permute(2, 0, 1).transpose(1, 2)
            
            # --- 聚焦一致性权重 ---
            # Focus Center: [N, 1]
            oc_center = focus_map[b].view(-1).unsqueeze(1)
            # Focus Neighbors: [N, K*K]
            oc_unfold = F.unfold(focus_map[b].unsqueeze(0), self.window_size, padding=pad)
            oc_neighbors = oc_unfold.view(1, self.window_size**2, N).permute(2, 0, 1).squeeze(0).transpose(0, 1)
            
            # Sim Calculation
            # [N, 1, C] @ [N, C, K*K] -> [N, 1, K*K]
            sim = torch.matmul(f_center, f_neighbors).squeeze(1)
            
            # Focus weighting: exp(-gamma * |fi - fj|)
            focus_diff = torch.abs(oc_center - oc_neighbors)
            weights = sim * torch.exp(-self.gamma * focus_diff)
            
            # Top-K
            _, local_idx = torch.topk(weights, k=min(self.k, self.window_size**2), dim=1)
            
            # Index Mapping (Local Window ID -> Global Pixel ID)
            # 生成全局网格 ID [1, 1, H, W]
            grid_ids = torch.arange(N, device=device).view(1, 1, H, W).float()
            ids_unfold = F.unfold(grid_ids, self.window_size, padding=pad)
            ids_neighbors = ids_unfold.view(1, self.window_size**2, N).permute(2, 0, 1).squeeze(0).transpose(0, 1).long()
            
            global_src = torch.arange(N, device=device).view(-1, 1).repeat(1, self.k).view(-1)
            global_dst = torch.gather(ids_neighbors, 1, local_idx).view(-1)
            
            # 处理 Batch 偏移
            batch_offset = b * N
            edge_indices_list.append(torch.stack([global_src + batch_offset, global_dst + batch_offset], dim=0))

        # 合并所有 Batch 的边
        edge_index = torch.cat(edge_indices_list, dim=1)
        return edge_index
class EfficientConvBlock(nn.Module):
    """
    创新功能2：高效特征提取模块
    结构：1x1 Exp -> 3x3 DW Conv -> 1x1 Proj + SE Attention (Optional) -> Residual
    """
    def __init__(self, in_channels, out_channels, expand_ratio=2):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        
        self.use_res_connect = (in_channels == out_channels)
        
        self.conv = nn.Sequential(
            # 1. Pointwise Conv (Expansion)
            nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            
            # 2. Depthwise Conv (Spatial Feature)
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            
            # 3. Pointwise Conv (Projection)
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
class FocusGNNFusionNet(nn.Module):
    def __init__(self, in_channels=3, feat_dim=64):
        super().__init__()
        
        # --- 1. Siamese Encoder (高效卷积) ---
        # 共享权重，用于提取 A 和 B 的特征
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.layer1 = EfficientConvBlock(32, feat_dim)
        self.layer2 = EfficientConvBlock(feat_dim, feat_dim)
        
        # --- 2. Graph Builder (创新点1) ---
        self.graph_builder = FocusAwareGraphBuilder(k=8, window_size=7, gamma=2.0)
        
        # --- 3. GCN Reasoner (图推理) ---
        # 输入特征维度是 feat_dim (因为我们将处理 feature difference)
        self.gcn1 = GCNConv(feat_dim, feat_dim)
        self.gcn2 = GCNConv(feat_dim, feat_dim)
        self.act = nn.ReLU(inplace=True)
        
        # --- 4. Decision Decoder ---
        # 将 CNN 特征和 GNN 推理结果融合生成 Mask
        self.decoder = nn.Sequential(
            EfficientConvBlock(feat_dim * 2, 32), # Concat(CNN_diff, GNN_out)
            nn.Conv2d(32, 1, 1) # 输出单通道 Mask
        )
        
        self.feat_dim = feat_dim

    def forward(self, img_a, img_b):
        """
        img_a, img_b: [B, 3, H, W]
        """
        B, C, H, W = img_a.shape
        N = H * W
        
        # 1. 提取特征 (Siamese Network)
        def extract_feat(x):
            x = self.stem(x)
            x = self.layer1(x)
            x = self.layer2(x)
            return x # [B, 64, H, W]
            
        feat_a = extract_feat(img_a)
        feat_b = extract_feat(img_b)
        
        # 2. 计算特征差值 (Decision Cues)
        # 融合网络的核心在于“比较”，差值特征包含了谁更清晰的信息
        # 比如：某处 feat_a 响应强烈，feat_b 响应微弱，diff 就很大
        feat_diff = torch.abs(feat_a - feat_b) # [B, 64, H, W]
        
        # 3. 构建聚焦感知图 (Innovation 1)
        # 我们基于 feat_diff 来构图，连接那些“差异模式相似”的节点
        edge_index = self.graph_builder(feat_diff)
        
        # 4. GNN 推理
        # 准备节点特征: [B*N, C]
        x_graph = feat_diff.permute(0, 2, 3, 1).reshape(B * N, -1)
        
        # GCN Layer 1
        x_graph = self.gcn1(x_graph, edge_index)
        x_graph = self.act(x_graph)
        # GCN Layer 2
        x_graph = self.gcn2(x_graph, edge_index)
        x_graph = self.act(x_graph) # [B*N, 64]
        
        # 5. 还原回 2D 空间
        feat_gnn = x_graph.view(B, H, W, self.feat_dim).permute(0, 3, 1, 2) # [B, 64, H, W]
        
        # 6. 融合与解码
        # 将原始的 CNN 差值特征与 GNN 修正后的特征拼接
        # CNN 提供局部细节，GNN 提供区域一致性
        feat_cat = torch.cat([feat_diff, feat_gnn], dim=1)
        
        logits = self.decoder(feat_cat) # [B, 1, H, W]
        decision_map = torch.sigmoid(logits) # 归一化到 0~1
        
        return decision_map

# ==========================================
# 完整性测试代码
# ==========================================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")

    # 1. 实例化模型
    model = FocusGNNFusionNet(in_channels=3, feat_dim=64).to(device)
    
    # 2. 创建模拟输入 (B=2, C=3, H=224, W=224)
    # 注意：B=2 可以测试 Batch 处理逻辑是否正确
    input_a = torch.randn(2, 3, 224, 224).to(device)
    input_b = torch.randn(2, 3, 224, 224).to(device)
    
    print("输入尺寸:", input_a.shape)
    
    # 3. 前向传播
    import time
    start_time = time.time()
    
    decision_map = model(input_a, input_b)
    
    end_time = time.time()
    
    print("输出决策图尺寸:", decision_map.shape) # 期望: [2, 1, 224, 224]
    print(f"推理耗时: {end_time - start_time:.4f}s")
    
    # 4. 显存占用检查 (如果是在 GPU 上)
    if torch.cuda.is_available():
        print(f"显存占用: {torch.cuda.max_memory_allocated() / 1024 / 1024:.2f} MB")

    # 5. 简单的验证
    assert decision_map.shape == (2, 1, 224, 224), "输出尺寸错误"
    assert decision_map.min() >= 0 and decision_map.max() <= 1, "输出未经过 Sigmoid 归一化"
    print("✅ 网络构建成功，逻辑测试通过。")