import os
import sys
import glob
import time
import cv2
import torch
from tqdm import tqdm
from torch import einsum
from Nets.SGUG2 import Network
from Utilities import Consistency
import Utilities.DataLoaderFM as DLr
from torch.utils.data import DataLoader
from Utilities.CUDA_Check import GPUorCPU

DEVICE = GPUorCPU.DEVICE


class ZeroOneNormalize(object):
    def __call__(self, img):
        return img.float().div(255)


class Fusion:
    def __init__(self,
                 modelpath='RunTimeData/best-model.ckpt',
                 dataroot='./Datasets/Eval',
                 dataset_names=None,  # 支持多个数据集列表
                 threshold=0.00,
                 window_size=5):
        """
        Args:
            dataset_names: 数据集名称列表，如 ['Lytro', 'MFFW', 'MFI-WHU', 'MFI']
                          如果为None，则自动检测所有数据集
        """
        self.DEVICE = GPUorCPU().DEVICE
        self.MODELPATH = modelpath
        self.DATAROOT = dataroot
        self.DATASET_NAMES = dataset_names if dataset_names is not None else self.get_available_datasets()
        self.THRESHOLD = threshold
        self.window_size = window_size
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
        print("=" * 80)
        print()

        # 加载模型（只需加载一次）
        MODEL = self.LoadWeights(self.MODELPATH)

        # 处理每个数据集
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
        model.load_state_dict(torch.load(modelpath))
        model.eval()
        
        # 尝试计算模型参数和FLOPs（使用较小的图像尺寸避免显存溢出）
        try:
            from thop import profile, clever_format
            # 使用256x256而非520x520，避免torch.cdist在大量节点时的显存溢出
            test_size = 256
            test_input_A = torch.rand(1, 3, test_size, test_size).to(self.DEVICE)
            test_input_B = torch.rand(1, 3, test_size, test_size).to(self.DEVICE)
            
            flops, params = profile(model, inputs=(test_input_A, test_input_B))
            flops, params = clever_format([flops, params], "%.5f")
            print('模型信息: flops: {}, params: {}'.format(flops, params))
        except Exception as e:
            # 如果profile失败（如显存不足），只打印参数数量
            print(f'警告: 无法计算FLOPs ({str(e)})')
            num_params = sum(p.numel() for p in model.parameters())
            print(f'模型参数数量: {num_params / 1e6:.2f} M')
        
        return model

    def PrepareData(self, datapath):
        eval_list_A = sorted(glob.glob(os.path.join(datapath, 'sourceA', '*.*')))
        eval_list_B = sorted(glob.glob(os.path.join(datapath, 'sourceB', '*.*')))
        print(f"找到 {len(eval_list_A)} 对图像")
        return eval_list_A, eval_list_B

    def ConsisVerif(self, img_tensor, threshold):
        # Verified_img_tensor = Consistency.Binarization(img_tensor)
        # if threshold != 0:
        Verified_img_tensor = Consistency.RemoveSmallArea(img_tensor=img_tensor, threshold=threshold)
        return Verified_img_tensor

    def FusionProcess(self, model, eval_list_A, eval_list_B, savepath, threshold, dataset_name):
        results_path = './Results' + savepath
        if not os.path.exists(results_path):
            os.makedirs(results_path, exist_ok=True)

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
                model_output = model(A, B)
                # 模型返回 (result, gates_a, gates_b)，只取result（决策图）
                D = model_output[0] if isinstance(model_output, tuple) else model_output
                D = torch.where(D > 0.5, 1., 0.)
                D = self.ConsisVerif(D, threshold)

                # 保存决策图（在转换为numpy之前）
                # D的形状: [batch, channels, height, width]，取第一个batch的第一个通道
                D_save = D[0, 0].clone().detach().cpu().numpy()  # 取单通道
                # 转换为0-255的uint8格式
                D_save = (D_save * 255).astype('uint8')
                decision_path = os.path.join(results_path, f'{dataset_name}map-{str(cnt).zfill(2)}.png')
                cv2.imwrite(decision_path, D_save)

                # 转换为numpy格式（用于融合）
                D = einsum('c w h -> w h c', D[0]).clone().detach().cpu().numpy()

                # 读取原始图像
                A_img = cv2.imread(eval_list_A[cnt - 1])
                B_img = cv2.imread(eval_list_B[cnt - 1])

                # 融合图像
                IniF = A_img * D + B_img * (1 - D)

                # 保存融合结果
                output_path = os.path.join(results_path, f'{dataset_name}-{str(cnt).zfill(2)}.png')
                cv2.imwrite(output_path, IniF)

                cnt += 1
                running_time.append(time.time() - start_time)

        # 统计时间
        running_time_total = 0
        for i in range(len(running_time)):
            if i != 0:  # 跳过第一次（可能包含模型加载时间）
                running_time_total += running_time[i]

        if len(running_time) > 1:
            avg_time = running_time_total / (len(running_time) - 1)
            print(f"\n[{dataset_name}] 平均处理时间: {avg_time:.4f} s")
            print(f"[{dataset_name}] 结果保存在: {results_path}\n")


if __name__ == '__main__':
    # 方式1: 指定数据集列表
    dataset_names = ['Lytro', 'MFFW', 'MFI', 'Grayscale']  # 修改此处指定要处理的数据集
    # dataset_names = ['Triple Series']
    # dataset_names = ['HBU']
    # 方式2: 自动检测所有数据集（推荐）
    # dataset_names = None  # 自动检测所有可用数据集

    f = Fusion(
        modelpath='./RunTimeData/2026-02-03 23.19.46/model34.ckpt',
        dataroot='./Datasets/Eval',
        dataset_names=dataset_names  # 可以是None（自动检测）或列表
    )
    f()
