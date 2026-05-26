import torch
from torch import nn
from thop import profile, clever_format
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm, GATConv, LayerNorm
import time
import sys
import os

# 确保引入了正确的采样函数
from Nets.AdaptiveSampling2 import FeatureSimilarityGraph_Adaptive as img_processes
# 确保引入了 AMoFE (如果你的目录下有这个文件)
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'AELF'))
from Nets.MOE import AMoFE

class GPUorCPU:
    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = 'cpu'
DEVICE = GPUorCPU.DEVICE

# ==========================================
# 基础组件 (Basic Modules)
# ==========================================

class SimAM(nn.Module):
    def __init__(self, lamda=1e-5):
        super().__init__()
        self.lamda = lamda
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        n = h * w - 1
        mean = torch.mean(x, dim=[-2, -1], keepdim=True)
        var = torch.sum(torch.pow((x - mean), 2), dim=[-2, -1], keepdim=True) / n
        e_t = torch.pow((x - mean), 2) / (4 * (var + self.lamda)) + 0.5
        out = self.sigmoid(e_t) * x
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

class conv3x3(nn.Module):
    "3x3 convolution with padding"
    def __init__(self, input_dim, output_dim, stride=1):
        super().__init__()
        self.att = SimAM()
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.Mish(),
        )

    def forward(self, x):
        x = self.att(x)
        out = self.conv3x3(x)
        return out

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Mish(inplace=True)
        )
    def forward(self, x):
        return self.body(x)

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.Mish(inplace=True)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class Initial_Conv(nn.Module):  
    def __init__(self, dim_in, out_channels):
        super(Initial_Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim_in, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, f1, f2):
        out1 = self.conv(f1)
        out2 = self.conv(f2)
        return out1, out2

class SELayer_2d(nn.Module):
    def __init__(self, channel, reduction):
        super(SELayer_2d, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.linear1 = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.Mish()
        )
        self.linear2 = nn.Sequential(
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, X_input):
        b, c, _, _ = X_input.size()
        y = self.avg_pool(X_input)
        y = y.view(b, c)
        y = self.linear1(y)
        y = self.linear2(y)
        y = y.view(b, c, 1, 1)
        return X_input * y.expand_as(X_input)

# ==========================================
# 【新增】空间注意力与求精头 (New Modules)
# ==========================================

class SpatialAttention(nn.Module):
    """
    【新增】空间注意力模块：利用 AvgPool 和 MaxPool 提取空间信息，
    强化连通区域的响应，帮助填补物体内部空洞。
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class RefinementHead(nn.Module):
    def __init__(self, in_channels=24):
        super(RefinementHead, self).__init__()
        # 多尺度膨胀卷积，专门对付多聚焦融合中的“孤立噪点”和“空洞”
        # 增加 padding 以保持尺寸一致：padding = dilation * (kernel_size - 1) // 2       
        # 分支1: 感受野 3x3
        self.dilated_conv1 = nn.Conv2d(in_channels, 16, 3, padding=2, dilation=2, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        # 分支2: 感受野更广，专门跨越较大的黑色空洞
        self.dilated_conv2 = nn.Conv2d(in_channels, 16, 3, padding=4, dilation=4, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        
        # 分支3: 1x1 保持原始细节
        self.conv1x1 = nn.Conv2d(in_channels, 16, 3, padding=1, dilation=1, bias=False)
        self.bn3 = nn.BatchNorm2d(16)
        
        self.act = nn.Mish(inplace=True)
        
        # 融合层：48 -> 16 -> 1 (决策图)
        self.fusion = nn.Sequential(
            nn.Conv2d(16*3, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.Mish(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1) # 最终映射为单通道决策图
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        d1 = self.act(self.bn1(self.dilated_conv1(x)))
        d2 = self.act(self.bn2(self.dilated_conv2(x)))
        c1 = self.act(self.bn3(self.conv1x1(x)))
        
        combined = torch.cat([d1, d2, c1], dim=1)
        out = self.fusion(combined)
        return self.sigmoid(out)

# ==========================================
# 核心网络组件 (Encoder / Decoder)
# ==========================================

class AELF_Encoder(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, act=None, bias=True, cross=True):
        super(AELF_Encoder, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.cross = cross
        
        if act is None:
            act = nn.Mish(inplace=True)
        
        n_feat = in_dim
        c1 = n_feat    
        c2 = 36        
        c3 = 48        
        
        self.encoder_level1 = ResBlock(c1, kernel_size, bias=bias, act=act)
        self.encoder_level2 = ResBlock(c2, kernel_size, bias=bias, act=act)
        self.encoder_level3 = ResBlock(c3, kernel_size, bias=bias, act=act)
        
        self.down12 = DownSample(c1, 36)
        self.down23 = DownSample(c2, 48)
        
        if cross:
            self.image_event_transformer1 = AMoFE(channels=c1, num_experts=3)
            self.image_event_transformer2 = AMoFE(channels=c2, num_experts=3)
            self.image_event_transformer3 = AMoFE(channels=c3, num_experts=3)
        
        self.output_conv = nn.Conv2d(c3, out_dim, kernel_size=1, bias=bias)
    
    def forward(self, x):
        enc1 = self.encoder_level1(x)
        gates1 = None
        if self.cross:
            enc1, gates1 = self.image_event_transformer1(enc1)
        x = self.down12(enc1)
        
        enc2 = self.encoder_level2(x)
        gates2 = None
        if self.cross:
            enc2, gates2 = self.image_event_transformer2(enc2)
        x = self.down23(enc2)
        
        enc3 = self.encoder_level3(x)
        gates3 = None
        if self.cross:
            enc3, gates3 = self.image_event_transformer3(enc3)
        
        out = self.output_conv(enc3)
        
        skips = [enc1, enc2] 
        
        if self.cross:
            return out, skips, [gates1, gates2, gates3]
        else:
            return out, skips, None

class AELF_Decoder(nn.Module):
    def __init__(self, in_dim, out_dim, enc_in_dim, kernel_size=3, act=None, bias=True):
        super(AELF_Decoder, self).__init__()
        
        if act is None:
            act = nn.Mish(inplace=True)
            
        c1 = enc_in_dim 
        c2 = 36         
        c3 = 48         
        
        self.init_conv = nn.Conv2d(in_dim, c3, kernel_size=1, bias=bias)
        self.body_level3 = ResBlock(c3, kernel_size, bias=bias, act=act)
        
        self.up23 = UpSample(c3, c2)
        self.fusion2 = nn.Sequential(
            nn.Conv2d(c2 * 2, c2, kernel_size=1, bias=bias),
            ResBlock(c2, kernel_size, bias=bias, act=act)
        )
        
        self.up12 = UpSample(c2, c1)
        self.fusion1 = nn.Sequential(
            nn.Conv2d(c1 * 2, c1, kernel_size=1, bias=bias),
            ResBlock(c1, kernel_size, bias=bias, act=act)
        )
        
        self.tail = nn.Conv2d(c1, out_dim, kernel_size=3, padding=1, bias=bias)

    def forward(self, x, skips):
        enc1, enc2 = skips
        
        x = self.init_conv(x)
        x = self.body_level3(x)
        
        x = self.up23(x)      
        if x.size() != enc2.size():
            x = F.interpolate(x, size=enc2.shape[-2:], mode='bilinear', align_corners=True)
            
        x = torch.cat([x, enc2], dim=1) 
        x = self.fusion2(x)            
        
        x = self.up12(x)      
        if x.size() != enc1.size():
            x = F.interpolate(x, size=enc1.shape[-2:], mode='bilinear', align_corners=True)
            
        x = torch.cat([x, enc1], dim=1) 
        x = self.fusion1(x)            
        
        out = self.tail(x)    
        return out

# ==========================================
# GCN 相关组件 (GCN Modules)
# ==========================================

class gcn_encoder(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, att_dim, out_dim):
        super(gcn_encoder, self).__init__()
        self.GConv1 = GCNConv(in_dim, hidden_dim)
        self.ln1 = LayerNorm(hidden_dim)
        self.at1 = nn.Mish()
        self.GATConv = GATConv(hidden_dim, att_dim, heads=3)
        self.ln2 = LayerNorm(att_dim * 3)
        self.at2 = nn.Mish()
        self.GConv2 = GCNConv(att_dim * 3, out_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.GConv1(x, edge_index)
        x = self.ln1(x)
        x = self.at1(x)
        out = x
        x = self.GATConv(x, edge_index)
        x = self.ln2(x)
        x = self.at2(x)
        gcn_result = self.GConv2(x, edge_index)
        return out, gcn_result

class gcn_decoder(nn.Module):
    # 修改 init，接收两个不同的维度参数
    def __init__(self, cnn_dim=24, gcn_dim=48, out_dim=24):
        super(gcn_decoder, self).__init__()
        
        # 自动计算拼接后的总维度
        mid_dim = cnn_dim + gcn_dim  
        
        self.block = nn.Sequential(
            # groups=mid_dim 保证它是 Depthwise Convolution
            nn.Conv2d(mid_dim, mid_dim, kernel_size=3, padding=1, groups=mid_dim, bias=False),
            nn.BatchNorm2d(mid_dim),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 后续压缩
            nn.Conv2d(mid_dim, gcn_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(gcn_dim),
            nn.Mish(inplace=True),
            
            nn.Conv2d(gcn_dim, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.Mish(inplace=True)
        )

    def forward(self, cnn_result, gcn_result):
        
        x = torch.cat([gcn_result, cnn_result], dim=1)
        return self.block(x)

# ==========================================
# 主网络 (Main Network)
# ==========================================

class Network(nn.Module):
    def __init__(self, cross=True, use_gcn=True):
        super(Network, self).__init__()
        self.cross = cross
        self.use_gcn = use_gcn
        
        self.feature_extraction = Initial_Conv(dim_in=3, out_channels=24)
        
        self.gcn_bn1 = nn.GroupNorm(6, 24)
        self.gcn_bn2 = nn.GroupNorm(6, 36)

        self.mix = nn.Conv2d(6, 1, kernel_size=7, stride=1, padding=3)

        # 1. 编码器
        self.cnn_encoder1 = AELF_Encoder(in_dim=24, out_dim=36, kernel_size=3, act=nn.Mish(inplace=True), bias=True, cross=cross)

        self.skip_mix_1 = nn.Conv2d(24 * 2, 24, kernel_size=1, bias=True)
        self.skip_mix_2 = nn.Conv2d(36 * 2, 36, kernel_size=1, bias=True)

        # 2. Decoder
        self.cnn_decoder = AELF_Decoder(in_dim=48, out_dim=24, enc_in_dim=24, kernel_size=3, act=nn.Mish(inplace=True), bias=True)

        self.fea_mix_cnn = conv3x3(input_dim=72, output_dim=48)
        self.fea_mix_gcn = conv3x3(input_dim=72, output_dim=48)
        
        # 3. GCN 模块
        if self.use_gcn:
            self.gcn_encoder = gcn_encoder(in_dim=3, hidden_dim=24, att_dim=36, out_dim=36)
            # 使用修改后的 gcn_decoder (带 SA)
            self.gcn_decoder = gcn_decoder(cnn_dim=24, gcn_dim=48, out_dim=24)
        else:
            self.gcn_decoder = None
        
        # 【修改】使用 RefinementHead 替代简单的 Conv+Sigmoid
        # 增强对空洞/噪点的平滑能力
        self.out = RefinementHead(in_channels=24)

    def forward(self, A, B):
        b, c, h, w = A.shape
        fea_a, fea_b = self.feature_extraction(A, B)
        fea = self.mix(torch.cat([A, B], dim=1))

        # ==================== GCN 处理 ====================
        if self.use_gcn:
            gcn_list_a1 = []
            gcn_list_b1 = []
            gcn_list_a2 = []
            gcn_list_b2 = []

            fea_list = torch.split(fea, 1, dim=0)
            fea_list_a = torch.split(A, 1, dim=0)
            fea_list_b = torch.split(B, 1, dim=0)

            # 动态调整 Grid Stride
            target_nodes = 6000 if h*w <= 512*512 else 9000
            grid_stride = max(2, int((h * w / target_nodes) ** 0.5))
            grid_stride = grid_stride + (grid_stride % 2)

            for i in range(b):
                gcn_data_a, gcn_data_b, trans_matrix = img_processes(
                    fea_list_a[i], fea_list_b[i], fea_list[i],
                    easy_stride=grid_stride,
                    return_trans_matrix=True
                )

                out_a, gcn_result_a = self.gcn_encoder(gcn_data_a)

                # 处理转换矩阵
                if trans_matrix is not None:
                    if trans_matrix.is_sparse:
                        dtype_orig = gcn_result_a.dtype
                        if trans_matrix.dtype != torch.float32:
                            trans_matrix_fp32 = trans_matrix.to(torch.float32)
                        else:
                            trans_matrix_fp32 = trans_matrix
                        
                        gcn_result_a_fp32 = gcn_result_a.float() if gcn_result_a.dtype != torch.float32 else gcn_result_a
                        out_a_fp32 = out_a.float() if out_a.dtype != torch.float32 else out_a
                        
                        device_type = 'cuda' if gcn_result_a.is_cuda else 'cpu'
                        with torch.autocast(enabled=False, device_type=device_type):
                            gcn_result_a = torch.sparse.mm(trans_matrix_fp32, gcn_result_a_fp32).to(dtype_orig)
                            out_a = torch.sparse.mm(trans_matrix_fp32, out_a_fp32).to(dtype_orig)
                    else:
                        gcn_result_a = torch.mm(trans_matrix, gcn_result_a)
                        out_a = torch.mm(trans_matrix, out_a)
                else:
                    h_s = h // grid_stride
                    w_s = w // grid_stride
                    gcn_result_a = gcn_result_a.permute(1, 0).view(1, 36, h_s, w_s)
                    gcn_result_a = F.interpolate(gcn_result_a, size=(h, w), mode='bilinear', align_corners=True)
                    out_a = out_a.permute(1, 0).view(1, 24, h_s, w_s)
                    out_a = F.interpolate(out_a, size=(h, w), mode='bilinear', align_corners=True)

                gcn_result_a = gcn_result_a.view(int(h), int(w), 36).permute(-1, 0, 1).unsqueeze(0)
                gcn_list_a1.append(gcn_result_a)
                out_a = out_a.view(h, w, 24).permute(-1, 0, 1).unsqueeze(0)
                gcn_list_a2.append(out_a)

                # 处理 B
                out_b, gcn_result_b = self.gcn_encoder(gcn_data_b)
                
                if trans_matrix is not None:
                    if trans_matrix.is_sparse:
                        dtype_orig = gcn_result_b.dtype
                        if trans_matrix.dtype != torch.float32:
                            trans_matrix_fp32 = trans_matrix.to(torch.float32)
                        else:
                            trans_matrix_fp32 = trans_matrix
                        
                        gcn_result_b_fp32 = gcn_result_b.float() if gcn_result_b.dtype != torch.float32 else gcn_result_b
                        out_b_fp32 = out_b.float() if out_b.dtype != torch.float32 else out_b
                        
                        device_type = 'cuda' if gcn_result_b.is_cuda else 'cpu'
                        with torch.autocast(enabled=False, device_type=device_type):
                            gcn_result_b = torch.sparse.mm(trans_matrix_fp32, gcn_result_b_fp32).to(dtype_orig)
                            out_b = torch.sparse.mm(trans_matrix_fp32, out_b_fp32).to(dtype_orig)
                    else:
                        gcn_result_b = torch.mm(trans_matrix, gcn_result_b)
                        out_b = torch.mm(trans_matrix, out_b)
                else:
                    h_s = h // grid_stride
                    w_s = w // grid_stride
                    gcn_result_b = gcn_result_b.permute(1, 0).view(1, 36, h_s, w_s)
                    gcn_result_b = F.interpolate(gcn_result_b, size=(h, w), mode='bilinear', align_corners=True)
                    out_b = out_b.permute(1, 0).view(1, 24, h_s, w_s)
                    out_b = F.interpolate(out_b, size=(h, w), mode='bilinear', align_corners=True)

                gcn_result_b = gcn_result_b.view(int(h), int(w), 36).permute(-1, 0, 1).unsqueeze(0)
                gcn_list_b1.append(gcn_result_b)
                out_b = out_b.view(h, w, 24).permute(-1, 0, 1).unsqueeze(0)
                gcn_list_b2.append(out_b)

            gcn_result_a = self.gcn_bn2(torch.cat(gcn_list_a1, dim=0))
            gcn_result_b = self.gcn_bn2(torch.cat(gcn_list_b1, dim=0))
            
            gcn_result = self.fea_mix_gcn(torch.cat([gcn_result_a, gcn_result_b], dim=1))

            out_a = self.gcn_bn1(torch.cat(gcn_list_a2, dim=0))
            out_b = self.gcn_bn1(torch.cat(gcn_list_b2, dim=0))

            fea_a = fea_a + out_b
            fea_b = fea_b + out_a
        else:
            gcn_result = None

        # 4. CNN 编码
        result_a, skips_a, gates_a = self.cnn_encoder1(fea_a)
        result_b, skips_b, gates_b = self.cnn_encoder1(fea_b)

        # 5. CNN 融合
        cnn_result = self.fea_mix_cnn(torch.cat([result_a, result_b], dim=1))

        # 6. Skip 融合
        skip1_fused = self.skip_mix_1(torch.cat([skips_a[0], skips_b[0]], dim=1))
        skip2_fused = self.skip_mix_2(torch.cat([skips_a[1], skips_b[1]], dim=1))
        fused_skips = [skip1_fused, skip2_fused]

        # 7. CNN 解码
        cnn_result = self.cnn_decoder(cnn_result, fused_skips)

        # 8. 最终融合 (CNN + GCN)
        if self.use_gcn:
            # 【修改】 GCN 特征平滑：减少稀疏采样插值带来的网格/高频噪声
            kernel = 3 if grid_stride <= 2 else 5
            gcn_result_smooth = F.avg_pool2d(
                gcn_result, kernel_size=kernel, stride=1, padding=kernel//2
            )
            # 融合
            result = self.gcn_decoder(cnn_result, gcn_result_smooth)
        else:
            result = cnn_result

        # 9. 输出层 (RefinementHead)
        result = self.out(result)

        return result, gates_a, gates_b


if __name__ == '__main__':
    test_tensor_A = torch.rand((2, 3, 224, 224)).to(DEVICE)
    test_tensor_B = torch.rand((2, 3, 224, 224)).to(DEVICE)
    model = Network(use_gcn=True).to(DEVICE)
    
    # Warm up
    _ = model(test_tensor_A, test_tensor_B)
    
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print("Model Parameters: {} M".format(round(num_params / 1e6, 4)))
    
    start_time = time.time()
    result, _, _ = model(test_tensor_A, test_tensor_B)
    print("Output shape:", result.shape)
    print(f"Inference time: {time.time() - start_time:.4f}s")


