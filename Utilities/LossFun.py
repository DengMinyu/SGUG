import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp

# ==========================================
# 1. 优化的 SSIM (移除 Variable, 增强兼容性)
# ==========================================
class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        # 将 window 注册为 buffer，这样保存模型时会带上，且会自动跟随 .to(device)
        self.register_buffer('window', self.create_window(window_size, self.channel))

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        window = window.to(img1.device)
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img1, img2):
        # 确保输入在 [0, 1] 范围 (SSIM 的假设)
        # 如果模型输出是 logits，这里应该先 sigmoid，但通常由 HybridLoss 控制
        return 1 - self._ssim(img1, img2, self.window, self.window_size, self.channel, self.size_average)

# ==========================================
# 2. 优化的 TV Loss (改为 L1 范数，保持边缘锐度)
# ==========================================
class L1_TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(L1_TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        
        # 使用 abs (L1) 而不是 pow(2) (L2)
        # L1 允许阶跃信号（锐利边缘），L2 会使边缘平滑模糊
        h_tv = torch.abs(x[:, :, 1:, :] - x[:, :, :h_x - 1, :]).sum()
        w_tv = torch.abs(x[:, :, :, 1:] - x[:, :, :, :w_x - 1]).sum()
        
        return self.TVLoss_weight * 2 * (h_tv + w_tv) / (batch_size * h_x * w_x)

# ==========================================
# 3. 修复的 Dice Loss (处理 Sigmoid 逻辑)
# ==========================================
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # 注意：这里假设输入的 inputs 已经是 [0,1] 的概率值
        # 如果输入是 Logits，请取消下面注释
        # inputs = torch.sigmoid(inputs) 
        
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (inputs_flat * targets_flat).sum()
        dice = (2. * intersection + smooth) / (inputs_flat.sum() + targets_flat.sum() + smooth)
        return 1 - dice

# ==========================================
# 4. 新增：Focal Loss (专注难分边缘样本)
# ==========================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCELoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

# ==========================================
# 5. 整合：Hybrid Loss Optimized
# ==========================================
class HybridLoss(nn.Module):
    def __init__(self):
        super(HybridLoss, self).__init__()
        # 使用 Focal Loss 替换标准 BCE，加强边缘学习
        self.focal = nn.L1Loss()
        # 也可以保留 BCE: self.bce = nn.BCELoss()
        
        self.ssim = SSIMLoss(window_size=11)
        self.dice = DiceLoss()
        self.tv = L1_TVLoss() # 使用 L1 TV

    def forward(self, pred, target):
        """
        重要假设：pred 已经是经过 Sigmoid 的概率值 [0, 1]
        如果网络输出层没有 Sigmoid，请在这里添加: pred = torch.sigmoid(pred)
        """
        
        # 1. 像素级: Focal Loss (比 BCE 更好处理类别不平衡和边缘)
        loss_pixel = self.focal(pred, target)
        
        # 2. 区域级: Dice Loss (保证形状一致性)
        loss_dice = self.dice(pred, target)
        
        # 3. 结构级: SSIM (填补空洞)
        loss_ssim = self.ssim(pred, target)

        # 4. 平滑级: TV Loss (去除噪点，保持边界锐利)
        loss_tv = self.tv(pred)

        # 组合权重建议：total_loss = 0.4 * loss_l1 + 0.6 * loss_ssim + 0.8 * loss_dice + 0.1 * loss_tv
        # SSIM 对空洞极其敏感，权重维持较高
        # Pixel (Focal/BCE) 是基础
        # TV 权重很小即可，只需抑制噪点
        # total_loss = 0.7 * loss_pixel + 0.5 * loss_ssim + 0.1 * loss_dice + 0.005 * loss_tv
        # total_loss = 0.7 * loss_pixel + 0.3 * loss_ssim 
        total_loss = (
        0.4 * loss_pixel +   # 局部准确
        0.35 * loss_ssim +   # 结构一致（核心）
        0.2 * loss_dice +    # 区域连通
        0.05 * loss_tv       # 去噪
            )
        
        return total_loss