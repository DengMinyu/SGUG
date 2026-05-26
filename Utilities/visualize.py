import os
import sys
import glob
import time
import cv2
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torch import einsum
from Nets.MECP import Network
from Utilities import Consistency
import Utilities.DataLoaderFM as DLr
from torch.utils.data import DataLoader
from Utilities.CUDA_Check import GPUorCPU

DEVICE = GPUorCPU.DEVICE


# ==========================================
#  新增：MoE Gate 可视化辅助函数 (优化版)
# ==========================================

def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def visualize_grid(images, titles=None, save_path="grid.png", cols=4, cmap='jet'):
    """辅助函数：将多张图拼成一个 Grid 保存"""
    n = len(images)
    rows = (n + cols - 1) // cols
    # 根据行列动态调整画布大小
    plt.figure(figsize=(cols * 3, rows * 3))
    
    for i, img in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        if titles:
            plt.title(titles[i], fontsize=10)
        plt.axis('off')
        # 如果是单通道(H,W)显示热力图，如果是(H,W,3)显示RGB
        if img.ndim == 2:
            plt.imshow(img, cmap=cmap, vmin=0, vmax=1)
        else:
            plt.imshow(img)
            
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

def save_cof_heatmaps(gates_list, imgs=None, save_dir='RunTimeData/cof_visual', 
                      prefix='sample', apply_softmax=False):
    """
    保存 Expert 权重热力图
    Args:
        gates_list: 模型返回的 gates，可以是 Tensor 或 Tensor 列表
        imgs: 原始输入图像 [B, C, H, W]，用于生成叠加图 (Overlay)
        apply_softmax: 如果 gates 是 logits，设为 True；如果是概率(0-1)，设为 False
    """
    if gates_list is None:
        return
    _ensure_dir(save_dir)

    if isinstance(gates_list, torch.Tensor):
        gates_list = [gates_list]

    # 1. 处理原始图片 (作为背景)
    img_bg = None
    H_img, W_img = 256, 256 # 默认备用尺寸
    
    if imgs is not None:
        # 取 Batch 第一个样本 [C, H, W]
        img_tensor = imgs[0].detach().cpu()
        H_img, W_img = img_tensor.shape[1], img_tensor.shape[2]
        
        # 简单反归一化用于显示 (假设输入在 0-1 或 0-255)
        img_arr = img_tensor.permute(1, 2, 0).numpy() # [H, W, C]
        img_min, img_max = img_arr.min(), img_arr.max()
        if img_max - img_min > 1e-9:
            img_bg = (img_arr - img_min) / (img_max - img_min)
        else:
            img_bg = img_arr

    # 2. 遍历每一层 (Level)
    for lvl_idx, gates in enumerate(gates_list):
        if gates is None or not isinstance(gates, torch.Tensor):
            continue
            
        # gates: [B, num_experts, H_l, W_l] -> 取第一个样本 [num_experts, H_l, W_l]
        gate_sample = gates[0].detach()
        
        # 数值处理
        if apply_softmax:
            gate_sample = F.softmax(gate_sample, dim=0)
        
        num_experts = gate_sample.shape[0]
        
        overlay_list = []
        title_list = []

        for e in range(num_experts):
            # 获取当前 Expert 的热力图
            g_map = gate_sample[e].float()
            
            # 上采样到原图大小 [1, 1, H, W]
            g_map_expanded = g_map.unsqueeze(0).unsqueeze(0)
            g_map_ups = F.interpolate(g_map_expanded, size=(H_img, W_img), 
                                      mode='bilinear', align_corners=False)
            g_map_ups = g_map_ups.squeeze().cpu().numpy() # [H_img, W_img]

            # 归一化策略：限制在 0-1 之间显示
            norm_map = np.clip(g_map_ups, 0, 1)
            
            # 生成叠加图 (Overlay)
            if img_bg is not None:
                cmap = plt.get_cmap('jet')
                heatmap_rgb = cmap(norm_map)[..., :3] # [H, W, 3]
                
                # Alpha Blending: 0.5 原图 + 0.5 热力图
                alpha = 0.5
                overlay = (1 - alpha) * img_bg + alpha * heatmap_rgb
                overlay = np.clip(overlay, 0, 1)
                overlay_list.append(overlay)
            else:
                # 如果没有原图，直接存热力图
                overlay_list.append(norm_map)
            
            title_list.append(f'Exp {e} (Max: {g_map_ups.max():.2f})')

        # 保存该层的所有 Experts 拼图
        if img_bg is not None:
            # 第一张放原图作为参考
            grid_imgs = [img_bg] + overlay_list
            grid_titles = ["Input Image"] + title_list
        else:
            grid_imgs = overlay_list
            grid_titles = title_list
            
        save_name = f"{prefix}_L{lvl_idx}_Experts.png"
        visualize_grid(grid_imgs, grid_titles, os.path.join(save_dir, save_name), cols=4)


# ==========================================
#  主类 Logic
# ==========================================

class ZeroOneNormalize(object):
    def __call__(self, img):
        return img.float().div(255)


class Fusion:
    def __init__(self,
                 modelpath='RunTimeData/best-model.ckpt',
                 dataroot='./Datasets/Eval',
                 dataset_names=None,
                 threshold=0.001,
                 window_size=5,
                 visualize_gates=False):  # <--- 新增参数：是否开启热力图可视化
        """
        Args:
            dataset_names: 数据集名称列表
            visualize_gates: 是否保存 MoE Gate 的可视化热力图 (会稍微降低运行速度)
        """
        self.DEVICE = GPUorCPU().DEVICE
        self.MODELPATH = modelpath
        self.DATAROOT = dataroot
        self.DATASET_NAMES = dataset_names if dataset_names is not None else self.get_available_datasets()
        self.THRESHOLD = threshold
        self.window_size = window_size
        self.visualize_gates = visualize_gates  # 保存设置
        self.window = torch.ones([1, 1, self.window_size, self.window_size], dtype=torch.float).to(self.DEVICE)

    def get_available_datasets(self):
        """自动检测所有可用的数据集"""
        datasets = []
        if os.path.exists(self.DATAROOT):
            for item in os.listdir(self.DATAROOT):
                dataset_path = os.path.join(self.DATAROOT, item)
                if os.path.isdir(dataset_path):
                    sourceA_path = os.path.join(dataset_path, 'sourceA')
                    sourceB_path = os.path.join(dataset_path, 'sourceB')
                    if os.path.exists(sourceA_path) and os.path.exists(sourceB_path):
                        datasets.append(item)
        return datasets

    def __call__(self, *args, **kwargs):
        if len(self.DATASET_NAMES) == 0:
            print("未找到任何数据集！请检查 dataroot 路径。")
            return

        print("=" * 80)
        print(f"开始处理 {len(self.DATASET_NAMES)} 个数据集: {self.DATASET_NAMES}")
        if self.visualize_gates:
            print("注意: 已开启 Gate 热力图可视化，结果将保存在 Results/.../gates_visual 中")
        print("=" * 80)
        print()

        # 加载模型
        MODEL = self.LoadWeights(self.MODELPATH)

        for dataset_name in self.DATASET_NAMES:
            print(f"\n{'='*80}")
            print(f"处理数据集: {dataset_name}")
            print(f"{'='*80}\n")

            self.DATASET_NAME = dataset_name
            self.SAVEPATH = '/' + dataset_name
            self.DATAPATH = self.DATAROOT + '/' + dataset_name

            EVAL_LIST_A, EVAL_LIST_B = self.PrepareData(self.DATAPATH)

            if len(EVAL_LIST_A) == 0 or len(EVAL_LIST_B) == 0:
                print(f"警告: 数据集 {dataset_name} 为空，跳过！\n")
                continue

            self.FusionProcess(MODEL, EVAL_LIST_A, EVAL_LIST_B, self.SAVEPATH, self.THRESHOLD, dataset_name)

        print("\n" + "=" * 80)
        print("所有数据集处理完成！")
        print("=" * 80)

    def LoadWeights(self, modelpath):
        model = Network().to(self.DEVICE)
        # 兼容性加载：strict=False 防止一些无关紧要的键值对不匹配报错
        model.load_state_dict(torch.load(modelpath, map_location=self.DEVICE), strict=False)
        model.eval()
        
        try:
            from thop import profile, clever_format
            test_size = 256
            test_input_A = torch.rand(1, 3, test_size, test_size).to(self.DEVICE)
            test_input_B = torch.rand(1, 3, test_size, test_size).to(self.DEVICE)
            
            flops, params = profile(model, inputs=(test_input_A, test_input_B), verbose=False)
            flops, params = clever_format([flops, params], "%.5f")
            print('模型信息: flops: {}, params: {}'.format(flops, params))
        except Exception as e:
            num_params = sum(p.numel() for p in model.parameters())
            print(f'模型参数数量: {num_params / 1e6:.2f} M (Profile Error: {e})')
        
        return model

    def PrepareData(self, datapath):
        eval_list_A = sorted(glob.glob(os.path.join(datapath, 'sourceA', '*.*')))
        eval_list_B = sorted(glob.glob(os.path.join(datapath, 'sourceB', '*.*')))
        print(f"找到 {len(eval_list_A)} 对图像")
        return eval_list_A, eval_list_B

    def ConsisVerif(self, img_tensor, threshold):
        # Verified_img_tensor = Consistency.Binarization(img_tensor)
        Verified_img_tensor = Consistency.RemoveSmallArea(img_tensor=img_tensor, threshold=threshold)
        return Verified_img_tensor

    def FusionProcess(self, model, eval_list_A, eval_list_B, savepath, threshold, dataset_name):
        results_path = './Results' + savepath
        
        # 结果保存路径
        if not os.path.exists(results_path):
            os.makedirs(results_path, exist_ok=True)
            
        # Gate 可视化保存路径
        vis_path = os.path.join(results_path, 'gates_visual')

        eval_data = DLr.Dataloader_Eval(eval_list_A, eval_list_B)
        eval_loader = DataLoader(dataset=eval_data, batch_size=1, shuffle=False)

        eval_loader_tqdm = tqdm(eval_loader, colour='blue', leave=True, file=sys.stdout,
                                desc=f"融合 [{dataset_name}]")

        cnt = 1
        running_time = []

        with torch.no_grad():
            for A, B in eval_loader_tqdm:
                start_time = time.time()

                # 模型推理
                A = A.to(self.DEVICE)
                B = B.to(self.DEVICE)
                
                model_output = model(A, B)
                
                # --- 解析输出 ---
                # 假设模型返回 (result, gates_list) 或 (result, gate_a, gate_b)
                # 这里做兼容性处理
                gates = None
                
                if isinstance(model_output, (list, tuple)):
                    D = model_output[0] # Decision map
                    if len(model_output) > 1:
                        # 尝试捕获剩下的作为 gates
                        # 如果 model_output[1] 是 list，直接用；如果是 tensor，转 list
                        raw_gates = model_output[1]
                        if isinstance(raw_gates, list):
                            gates = raw_gates
                        elif isinstance(raw_gates, torch.Tensor):
                            gates = [raw_gates]
                else:
                    D = model_output
                    gates = None

                # --- 核心处理 ---
                D = torch.where(D > 0.5, 1., 0.)
                D = self.ConsisVerif(D, threshold)

                # --- Gate 可视化 (整合点) ---
                if self.visualize_gates and gates is not None:
                    # 使用源图像 A 作为热力图叠加的背景
                    save_cof_heatmaps(
                        gates_list=gates,
                        imgs=A, # 传入原图 tensor [B, C, H, W]
                        save_dir=vis_path,
                        prefix=f"{dataset_name}_{str(cnt).zfill(2)}",
                        apply_softmax=False # 如果你的 gate 输出还没过激活函数，请改为 True
                    )

                # --- 保存决策图 ---
                D_save = D[0, 0].clone().detach().cpu().numpy()
                D_save = (D_save * 255).astype('uint8')
                decision_path = os.path.join(results_path, f'{dataset_name}map-{str(cnt).zfill(2)}.png')
                cv2.imwrite(decision_path, D_save)

                # --- 融合计算 ---
                D_numpy = einsum('c w h -> w h c', D[0]).clone().detach().cpu().numpy()
                
                # 重新读取原始图像 (确保色彩空间一致)
                A_img = cv2.imread(eval_list_A[cnt - 1])
                B_img = cv2.imread(eval_list_B[cnt - 1])
                
                # 简单的线性融合
                IniF = A_img * D_numpy + B_img * (1 - D_numpy)

                output_path = os.path.join(results_path, f'{dataset_name}-{str(cnt).zfill(2)}.png')
                cv2.imwrite(output_path, IniF)

                cnt += 1
                running_time.append(time.time() - start_time)

        # 统计时间
        running_time_total = 0
        for i in range(len(running_time)):
            if i != 0:
                running_time_total += running_time[i]

        if len(running_time) > 1:
            avg_time = running_time_total / (len(running_time) - 1)
            print(f"\n[{dataset_name}] 平均处理时间: {avg_time:.4f} s")
            print(f"[{dataset_name}] 结果保存在: {results_path}\n")


if __name__ == '__main__':
    # 方式1: 指定数据集列表
    dataset_names = ['Lytro', 'MFFW', 'MFI-WHU', 'MFD']
    
    # 方式2: 自动检测
    # dataset_names = None

    f = Fusion(
        modelpath='./RunTimeData/2026-01-20 17.27.23/best_network.pth',
        dataroot='./Datasets/Eval',
        dataset_names=dataset_names,
        # ---------------------------
        # 在这里开启可视化开关
        # ---------------------------
        visualize_gates=True  # 设置为 True 即可生成热力图
    )
    f()