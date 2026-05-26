import torch
import torch.nn as nn
import torch.nn.functional as F

# PyTorch Geometric (PyG) 依赖
# 用于图卷积 (GCN), 图注意力 (GAT) 和 图归一化
from torch_geometric.nn import GCNConv, GATConv, LayerNorm

# --- 关键提示 ---
# 1. 还需要确保引入了 ResBlock 类 (在你之前的代码块中定义过)
#    如果 ResBlock 在另一个文件，需要: from YourModelFile import ResBlock
# 2. 还需要引入自适应采样函数 img_processes
#    根据你之前的路径，通常是这样:
from Nets.AdaptiveSampling2 import FeatureSimilarityGraph_Adaptive as img_processes
class ImprovedGCNEncoder(nn.Module):
    """
    优化后的 GCN 编码器：
    1. 增加 Residual 连接
    2. 混合 GCN (局部) 和 GAT (全局注意力)
    """
    def __init__(self, in_dim, hidden_dim, out_dim, heads=4):
        super(ImprovedGCNEncoder, self).__init__()
        
        # 第一层 GCN
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.bn1 = LayerNorm(hidden_dim)
        self.act1 = nn.Mish(inplace=True)
        
        # 中间层 GAT (引入注意力机制)
        self.conv2 = GATConv(hidden_dim, hidden_dim // heads, heads=heads, concat=True)
        self.bn2 = LayerNorm(hidden_dim)
        self.act2 = nn.Mish(inplace=True)
        
        # 第三层 GCN (映射回输出维度)
        self.conv3 = GCNConv(hidden_dim, out_dim)
        self.bn3 = LayerNorm(out_dim)
        self.act3 = nn.Mish(inplace=True)

        # 降维/对齐用的 1x1 卷积 (用于残差连接)
        self.shortcut = nn.Linear(in_dim, hidden_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # --- Stage 1 ---
        input_x = x
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.act1(x)
        
        # --- Stage 2 (GAT with Residual) ---
        # 如果维度不匹配，投影 input_x
        if input_x.shape[1] != x.shape[1]:
            res = self.shortcut(input_x)
        else:
            res = input_x
        
        # 残差连接：加上上一层的特征 (防止特征丢失)
        x = x + res  
        
        # GAT 处理
        x_gat = self.conv2(x, edge_index)
        x_gat = self.bn2(x_gat)
        x_gat = self.act2(x_gat)
        x = x + x_gat # Residual Again

        # --- Stage 3 ---
        out = self.conv3(x, edge_index)
        # out = self.bn3(out) # 最后一层通常不加激活，或者看任务需求
        
        return out

class ResBlock(nn.Module):
    def __init__(self, n_feat, kernel_size=3, bias=True, act=None):
        super(ResBlock, self).__init__()
        if act is None:
            act = nn.Mish(inplace=True)
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size, padding=1, bias=bias),
            nn.BatchNorm2d(n_feat),
            act,
            nn.Conv2d(n_feat, n_feat, kernel_size, padding=1, bias=bias),
            nn.BatchNorm2d(n_feat)
        )

    def forward(self, x):
        res = self.body(x)
        res += x
        return res
class RefinementHead(nn.Module):
    def __init__(self, in_channels):
        super(RefinementHead, self).__init__()
        # 1. 大感受野分支：膨胀卷积 (Dilation=2, 4) 用于填补空洞
        self.dilated_conv1 = nn.Conv2d(in_channels, 16, 3, padding=2, dilation=2)
        self.dilated_conv2 = nn.Conv2d(in_channels, 16, 3, padding=4, dilation=4)
        self.conv1x1 = nn.Conv2d(in_channels, 16, 1)
        
        self.fusion = nn.Sequential(
            nn.Conv2d(16*3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1) # 最终输出
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        d1 = self.dilated_conv1(x)
        d2 = self.dilated_conv2(x)
        c1 = self.conv1x1(x)
        out = torch.cat([d1, d2, c1], dim=1)
        out = self.fusion(out)
        return self.sigmoid(out)


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        
        # 1. 增强的特征提取器 (Mini Backbone)
        # 使用 ResBlock 替代原来的 Initial_Conv，提取更扎实的特征
        self.pre_conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.Mish()
        )
        # 两个 ResBlock 增加深度
        # self.feature_encoder = nn.Sequential(
        #     ResBlock(16, kernel_size=3),
        #     ResBlock(16, kernel_size=3)
        # )
        
        # 混合 A 和 B 的特征用于构图 (Metric Learning)
        self.mix_conv = nn.Conv2d(16 * 2, 16, kernel_size=3, padding=1)

        # 2. 改进的 GCN 编码器
        # 输入 16 -> 隐藏 64 -> 输出 32
        self.gcn_encoder = ImprovedGCNEncoder(in_dim=16, hidden_dim=64, out_dim=32)
        
        # 3. 后处理融合 (Refinement)
        # 输入维度 = (GCN_A: 32) + (GCN_B: 32) + (CNN_A: 16) + (CNN_B: 16) = 96
        # 引入 CNN 特征 (Skip Connection) 是关键！
        self.decoder = RefinementHead(96)

    def forward(self, A, B):
            b, c, h, w = A.shape
            DEVICE = A.device
    
            # --- 1. 特征提取 (CNN) ---
            fea_a = self.pre_conv(A) 
            fea_b = self.pre_conv(B)
            
            # fea_a = self.feature_encoder(fea_a_raw) 
            # fea_b = self.feature_encoder(fea_b_raw)
            
            fea_mix = self.mix_conv(torch.cat([fea_a, fea_b], dim=1))
    
            # --- 2. 构图与 GCN ---
            max_nodes = 4096 
            min_grid_stride = max(2, int((h * w / max_nodes) ** 0.5))
            grid_stride = ((min_grid_stride + 1) // 2) * 2
    
            data_a, data_b, trans_matrix = img_processes(
                img_a=fea_a, 
                img_b=fea_b, 
                fea_img=fea_mix,
                easy_stride=grid_stride,
                return_trans_matrix=True
            )
    
            # --- 3. GCN 推理 ---
            gcn_out_a = self.gcn_encoder(data_a) 
            gcn_out_b = self.gcn_encoder(data_b) 
    
            # --- 4. 投影回像素空间 (关键修改：强制 Float32) ---
            if trans_matrix is not None:
                # 1. 记录原始数据类型 (可能是 float16)
                orig_dtype = gcn_out_a.dtype 
    
                # 2. 强制转换为 float32 (解决 "not implemented for Half" 报错)
                # 稀疏矩阵和稠密特征都必须是 float32
                trans_matrix_32 = trans_matrix.to(torch.float32)
                gcn_out_a_32 = gcn_out_a.to(torch.float32)
                gcn_out_b_32 = gcn_out_b.to(torch.float32)
    
                # 3. 执行矩阵乘法 (在 float32 下进行)
                # 暂时禁用 autocast，防止它自动把 float32 又转回 float16
                with torch.autocast(device_type='cuda', enabled=False):
                    gcn_pixel_a = torch.mm(trans_matrix_32, gcn_out_a_32)
                    gcn_pixel_b = torch.mm(trans_matrix_32, gcn_out_b_32)
                
                # 4. 转回原始类型 (如 float16) 以匹配后续网络
                gcn_pixel_a = gcn_pixel_a.to(orig_dtype)
                gcn_pixel_b = gcn_pixel_b.to(orig_dtype)
                
                # Reshape 回图片尺寸
                gcn_pixel_a = gcn_pixel_a.view(b, h, w, -1).permute(0, 3, 1, 2)
                gcn_pixel_b = gcn_pixel_b.view(b, h, w, -1).permute(0, 3, 1, 2)
            else:
                # Fallback
                gcn_pixel_a = F.interpolate(gcn_out_a.view(b, -1, h//grid_stride, w//grid_stride), size=(h,w))
                gcn_pixel_b = F.interpolate(gcn_out_b.view(b, -1, h//grid_stride, w//grid_stride), size=(h,w))
    
            # --- 5. 融合与解码 ---
            cat_feat = torch.cat([
                gcn_pixel_a,   
                gcn_pixel_b,   
                fea_a,         
                fea_b          
            ], dim=1)
    
            out = self.decoder(cat_feat)
            
            return out, None, None