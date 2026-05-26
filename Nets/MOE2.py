import torch
import numpy as np
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import init as init

# ==========================================
# Basic Blocks (保持原样或微调)
# ==========================================
def conv(in_channels, out_channels, kernel_size, bias=False, stride=1, padding=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=padding, bias=bias, stride=stride)

class ResBlock(nn.Module):
    def __init__(self, n_feat, kernel_size, bias, act):
        super(ResBlock, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class ContextAwareNoisySpatialGate(nn.Module):

    def __init__(self, input_size, num_experts, top_k, noise_strength=0.1, use_temperature_annealing=True):
        super(ContextAwareNoisySpatialGate, self).__init__()

        self.input_size = input_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_strength = noise_strength
        self.use_temperature_annealing = use_temperature_annealing
        
        # --- 温度控制 ---
        self.register_buffer('temperature', torch.tensor(1.0))

        # ============================================================
        # 1. 全局上下文分支 (Global Context Branch) - 对应原代码的fc0
        # ============================================================
        # 保持你原来的双池化设计，捕捉全局信息
        self.pool_max = nn.AdaptiveMaxPool2d(1)
        self.pool_avg = nn.AdaptiveAvgPool2d(1)
        
        # 将原来的 fc0 拆分为两层，增加非线性能力，提取全局偏置
        self.global_fc = nn.Sequential(
            nn.Linear(input_size, input_size // 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(input_size // 4, num_experts)
        )

        # ============================================================
        # 2. 局部空间分支 (Local Spatial Branch) - 新增
        # ============================================================
        # 使用卷积提取像素级特征，保留空间分辨率 [H, W]
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(input_size, input_size // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(input_size // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(input_size // 2, num_experts, kernel_size=1)
        )
        
        # ============================================================
        # 3. 噪声生成器 (Noise Generator) - 对应原代码的fc1
        # ============================================================
        # 使用1x1卷积在每个像素生成独立的噪声参数
        self.noise_conv = nn.Conv2d(input_size, num_experts, kernel_size=1)
        self.sp = nn.Softplus()
        
        # 初始化
        self.softmax = nn.Softmax(dim=1)
        self._init_weights()

    def _init_weights(self):
        # 初始化噪声层为0，确保初始状态下噪声主要由 Softplus 控制
        init.zeros_(self.noise_conv.weight)
        if self.noise_conv.bias is not None:
            init.zeros_(self.noise_conv.bias)
        
        # 可选：初始化 spatial_conv 的最后一层为 0，
        # 让训练初期主要依赖全局上下文，避免局部过拟合
        init.zeros_(self.spatial_conv[-1].weight)
        if self.spatial_conv[-1].bias is not None:
            init.zeros_(self.spatial_conv[-1].bias)

    def forward(self, x):
        """
        Input: [B, C, H, W]
        Output: [B, Num_Experts, H, W]
        """
        B, C, H, W = x.shape
        
        # ---------------------------------------------------
        # Step A: 计算 Logits (信号部分)
        # ---------------------------------------------------
        
        # 1. 计算全局上下文 Logits (Global Context)
        # Global Pooling: [B, C, H, W] -> [B, C, 1, 1]
        global_feat = self.pool_max(x) + self.pool_avg(x)
        global_feat = global_feat.view(B, -1) # Flatten -> [B, C]
        # FC: [B, Num_Experts] -> View -> [B, Num_Experts, 1, 1] 以便广播
        global_logits = self.global_fc(global_feat).view(B, self.num_experts, 1, 1)
        
        # 2. 计算局部空间 Logits (Local Spatial)
        # Conv: [B, C, H, W] -> [B, Num_Experts, H, W]
        local_logits = self.spatial_conv(x)
        
        # 3. 融合 (Context-Aware)
        # 局部细节 + 全局偏置
        clean_logits = local_logits + global_logits
        
        # ---------------------------------------------------
        # Step B: 加入噪声 (Noisy Gating) - 严格复刻原代码逻辑
        # ---------------------------------------------------
        if self.training and self.noise_strength > 0:
            # 生成噪声的标准差参数: [B, Num_Experts, H, W]
            raw_noise_std = self.noise_conv(x)
            noise_std = self.sp(raw_noise_std)
            
            # 采样标准正态分布
            noise = torch.randn_like(clean_logits) * noise_std
            
            # --- 关键：原代码的噪声归一化逻辑适配到 4D Tensor ---
            # noise_mean 计算维度 dim=1 (Experts维度)，保持 keepdim=True 以便广播
            noise_mean = torch.mean(noise, dim=1, keepdim=True) # [B, 1, H, W]
            std = torch.std(noise, dim=1, keepdim=True)         # [B, 1, H, W]
            
            # 归一化噪声
            norm_noise = (noise - noise_mean) / (std + 1e-6) # 加 eps 防除零
            
            # 最终 Logits
            final_logits = clean_logits + (norm_noise * self.noise_strength)
        else:
            final_logits = clean_logits

        # ---------------------------------------------------
        # Step C: Top-K 选择 (Sparse Gating)
        # ---------------------------------------------------
        # 在 dim=1 (Expert维度) 上选取 Top-K
        # values: [B, K, H, W], indices: [B, K, H, W]
        top_k_logits, top_k_indices = torch.topk(final_logits, k=self.top_k, dim=1)
        
        # 创建遮罩，默认全为 -inf
        mask = torch.full_like(final_logits, float('-inf'))
        
        # 使用 scatter 将 top_k 的值填回原位置
        # 结果: 只有 Top-K 的位置有值，其余位置全是 -inf
        mask.scatter_(dim=1, index=top_k_indices, src=top_k_logits)
        
        # ---------------------------------------------------
        # Step D: 温度退火与 Softmax
        # ---------------------------------------------------
        # Softmax(-inf) 会变成 0，从而实现稀疏激活
        if self.use_temperature_annealing:
            temp = torch.clamp(self.temperature, min=1e-3)
            gate = self.softmax(mask / temp)
        else:
            gate = self.softmax(mask)

        return gate

    def update_temperature(self, new_temp):
        self.temperature.fill_(new_temp)
class EnhancedSpatialGate(nn.Module):

    def __init__(self, input_size, num_experts, top_k, noise_strength=0.1, use_temperature_annealing=True):
        super(EnhancedSpatialGate, self).__init__()

        self.input_size = input_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_strength = noise_strength
        self.use_temperature_annealing = use_temperature_annealing
        
        self.register_buffer('temperature', torch.tensor(1.0))

        # ============================================================
        # 1. 全局上下文分支 (Global Context / SE-Block Style)
        # ============================================================
        self.pool_max = nn.AdaptiveMaxPool2d(1)
        self.pool_avg = nn.AdaptiveAvgPool2d(1)
        
        # 这里的 FC 不仅输出 Logits 偏置，还可以用于通道注意力(可选)，这里主要用于生成全局 Bias
        self.global_fc = nn.Sequential(
            nn.Linear(input_size, input_size // 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(input_size // 4, num_experts)
        )

        # ============================================================
        # 2. 局部空间分支 (Local Spatial Branch) - [优化核心]
        # ============================================================
        # [优化] 使用空洞卷积扩大感受野，更容易区分平滑背景和聚焦纹理
        # [优化] 移除 BatchNorm，改用 GroupNorm 或不使用
        self.spatial_conv = nn.Sequential(
            # Layer 1: 空洞卷积 (dilation=2)，感受野 5x5
            nn.Conv2d(input_size, input_size // 2, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.GroupNorm(4, input_size // 2), # 使用 GroupNorm 对 BatchSize 不敏感，且保护对比度
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 2: 标准卷积
            nn.Conv2d(input_size // 2, num_experts, kernel_size=1)
        )
        
        # ============================================================
        # 3. 噪声生成器
        # ============================================================
        self.noise_conv = nn.Conv2d(input_size, num_experts, kernel_size=1)
        self.sp = nn.Softplus()
        
        self.softmax = nn.Softmax(dim=1)
        self._init_weights()

    def _init_weights(self):
        # 噪声层初始化为0
        init.zeros_(self.noise_conv.weight)
        if self.noise_conv.bias is not None:
            init.zeros_(self.noise_conv.bias)
        
        # [关键] 初始化 spatial_conv 最后一层为 0
        # 这样初始状态下 Logits 主要由 global_bias 决定（先验概率），防止初始的随机局部纹理导致专家选择震荡
        init.zeros_(self.spatial_conv[-1].weight)
        if self.spatial_conv[-1].bias is not None:
            init.zeros_(self.spatial_conv[-1].bias)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # ---------------------------------------------------
        # Step A: 全局上下文 (Global Prior)
        # ---------------------------------------------------
        # [优化] 融合 Max 和 Avg 特征
        global_feat = self.pool_max(x) + self.pool_avg(x)
        global_feat = global_feat.view(B, -1)
        # [B, Num_Experts, 1, 1]
        global_logits = self.global_fc(global_feat).view(B, self.num_experts, 1, 1)
        
        # ---------------------------------------------------
        # Step B: 局部空间特征 (Local Texture)
        # ---------------------------------------------------
        # [B, Num_Experts, H, W]
        local_logits = self.spatial_conv(x)
        
        # 融合: 局部判别 + 全局先验
        clean_logits = local_logits + global_logits
        
        # ---------------------------------------------------
        # Step C: 标准 MoE 噪声机制 (Standard Noisy Gating)
        # ---------------------------------------------------
        if self.training and self.noise_strength > 0:
            # 1. 计算噪声强度 sigma = Softplus(Wx)
            raw_noise_std = self.noise_conv(x)
            noise_std = self.sp(raw_noise_std)
            
            # 2. 生成标准正态分布噪声 epsilon ~ N(0, 1)
            epsilon = torch.randn_like(clean_logits)
            
            # 3. 加噪: Logits + sigma * epsilon
            # [优化] 移除了原来不稳定的 (noise-mean)/std 归一化
            # 因为 epsilon 本身就是标准的，直接乘强度即可，数学上更严谨且数值更稳定
            noisy_logits = clean_logits + (noise_std * epsilon * self.noise_strength)
        else:
            noisy_logits = clean_logits

        # ---------------------------------------------------
        # Step D: Top-K 稀疏选择
        # ---------------------------------------------------
        top_k_logits, top_k_indices = torch.topk(noisy_logits, k=self.top_k, dim=1)
        
        mask = torch.full_like(noisy_logits, float('-inf'))
        mask.scatter_(dim=1, index=top_k_indices, src=top_k_logits)
        
        # ---------------------------------------------------
        # Step E: 温度退火
        # ---------------------------------------------------
        if self.use_temperature_annealing:
            temp = torch.clamp(self.temperature, min=1e-3)
            gate = self.softmax(mask / temp)
        else:
            gate = self.softmax(mask)

        return gate

    def update_temperature(self, new_temp):
        self.temperature.fill_(new_temp)
# 专家 0: 空间专家 (保持不变，用于基础特征提取)
class SpatialExpert(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        res = x
        x = F.leaky_relu(self.conv1(x))
        attn = self.spatial_attention(x)
        x = x * attn
        x = self.conv2(x)
        return x + res # 添加残差连接更稳定

# 【创新改进】专家 1: 真·频域专家 (Real FFT Spectral Expert)
# 使用快速傅里叶变换处理全局频域信息
class RealSpectralExpert(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 定义一个可学习的复数权重参数
        # 初始值设得很小，进行残差学习
        self.scale = 0.02
        # 频域不仅有幅值还有相位，使用复数权重
        self.complex_weight = nn.Parameter(torch.randn(channels, 2, dtype=torch.float32) * self.scale)

    def forward(self, x):
        B, C, H, W = x.shape
        # 为了避免 cuFFT 在 half precision 下对尺寸的限制，强制在 float32 上执行 FFT
        dtype_orig = x.dtype
        device = x.device
        x_fp32 = x.float() if x.dtype != torch.float32 else x

        # 1. 转到频域 (Real FFT2) — 使用 float32
        x_fft = torch.fft.rfft2(x_fp32, norm='backward')

        # 2. 频域滤波/调制
        # 将参数转换为复数视图: [C, 2] -> [C] (complex)
        weight = torch.view_as_complex(self.complex_weight)
        weight = weight.to(x_fft.dtype)
        # 调整形状以便广播: [C, 1, 1]
        weight = weight.view(C, 1, 1)

        # 在频域进行点乘操作 (相当于空域的全局大核卷积)
        x_fft_modulated = x_fft * weight

        # 3. 转回空域 (Inverse Real FFT2)
        out = torch.fft.irfft2(x_fft_modulated, s=(H, W), norm='backward')

        # 恢复为输入的原始 dtype（如果需要）
        if dtype_orig != torch.float32:
            out = out.to(dtype=dtype_orig)

        return out + x_fp32.to(out.dtype) # 残差连接

# 【性能优化】专家 2: 优化版边缘专家 (Optimized Edge Expert)
class OptimizedEdgeExpert(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 使用分组卷积作为边缘提取器，初始化为拉普拉斯算子
        self.edge_extract = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        
        # 【优化】向量化初始化，移除慢速 for 循环
        laplacian_kernel = torch.tensor([[-1., -1., -1.], 
                                         [-1., 8., -1.], 
                                         [-1., -1., -1.]])
        # 调整形状为 [1, 1, 3, 3] 并重复 channels 次 -> [C, 1, 3, 3]
        laplacian_kernel = laplacian_kernel.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        
        # 冻结梯度 (可选：如果你希望它是一个固定的拉普拉斯算子)
        # 或者允许梯度 (如下)，让网络微调边缘提取方式
        with torch.no_grad():
            self.edge_extract.weight.copy_(laplacian_kernel)
            
        # 后续融合层
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        # 提取边缘 (不再强制取 abs，允许网络学习负响应)
        edge = self.edge_extract(x)
        # 特征变换
        edge_fusion = self.act(self.conv1(edge))
        return edge_fusion + x # 残差连接

# ==========================================
# Main Module: Spatial-Aware AMoFE
# ==========================================
class AMoFE(nn.Module):

    def __init__(self, channels, num_experts=3, temperature=1.0, top_k=2):
        super(AMoFE, self).__init__()
        
        # ============================================================
        # 修改点：替换为 ContextAwareNoisySpatialGate
        # ============================================================
        self.gate = EnhancedSpatialGate(
            input_size=channels,
            num_experts=num_experts,
            top_k=top_k,               # 核心参数：从3个专家中选出最强的k个
            noise_strength=0.1,        # 噪声强度，用于负载均衡和鲁棒性
            use_temperature_annealing=True
        )
        
        # 立即同步初始温度 (因为新Gate内部默认初始化为1.0)
        self.gate.update_temperature(temperature)
 
        # 初始化专家列表 (保持不变)
        self.expert_networks_d = nn.ModuleList([
            SpatialExpert(channels),          # Expert 0: 局部空间
            RealSpectralExpert(channels),     # Expert 1: 全局频域
            OptimizedEdgeExpert(channels),    # Expert 2: 高频边缘
        ])
        
        assert len(self.expert_networks_d) == num_experts
        self.num_experts = num_experts

    def forward(self, x_):
        # 计算门控权重
        # 输出: [B, Num_Experts, H, W]
        # 注意：由于使用了Top-K，未被选中的专家权重此时已经是 0
        cof = self.gate(x_)
        
        # 初始化输出张量
        out = torch.zeros_like(x_)

        # 混合专家输出
        for idx in range(self.num_experts):
            # 获取当前专家的空间权重: [B, H, W]
            # 并增加一个通道维度以便广播: [B, 1, H, W]
            coef_spatial = cof[:, idx, :, :].unsqueeze(1)
            
            # 优化：如果该专家在整个Batch中权重全为0（极其罕见但可能），可跳过计算节省算力
            # if coef_spatial.sum() == 0: continue 

            # 计算当前专家输出
            expert_out = self.expert_networks_d[idx](x_)  
            
            # 像素级加权累加
            out += expert_out * coef_spatial

        return out, cof

# ==========================================
# 测试 Demo
# ==========================================
if __name__ == '__main__':
    # 假设输入参数
    batch_size = 2
    channels = 64
    height, width = 128, 128
    
    # 创建模型实例
    model = AMoFE(channels=channels, num_experts=3)
    
    # 创建模拟输入数据 [B, C, H, W]
    dummy_input = torch.randn(batch_size, channels, height, width)
    
    print("输入形状:", dummy_input.shape)
    
    # 前向传播
    output, gate_weights = model(dummy_input)
    
    print("输出形状:", output.shape) # 应该与输入相同 [2, 64, 128, 128]
    # 关键验证：门控权重的形状应该是 [B, Num_Experts, H, W]
    print("空间门控权重形状:", gate_weights.shape) # 应该是 [2, 3, 128, 128]
    
    # 验证某个像素点的权重和是否为 1
    pixel_sum = torch.sum(gate_weights[0, :, 64, 64])
    print(f"中心像素 (64,64) 的专家权重之和: {pixel_sum.item():.4f}") # 应该接近 1.0

    print("\n模型结构验证成功，已集成空间门控与新专家模块。")
