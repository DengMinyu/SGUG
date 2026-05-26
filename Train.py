import os
import sys
import glob
import torch
import datetime
from torch import nn
from tqdm import tqdm
from Nets.SGUG2 import Network
import Utilities.DataLoaderFM as DLr
from torch.utils.data import DataLoader
from Utilities.CUDA_Check import GPUorCPU
from Utilities.LossFun import HybridLoss
from Utilities.Logging_SaveModel import Logging_SaveModel
from Utilities.visualize import save_cof_heatmaps
DEVICE = GPUorCPU.DEVICE
import matplotlib.pyplot as plt


class NetTrain:
    def __init__(self,
                 # data_path='../../lanyun-tmp/MECPData',
                 data_path='./Datasets/Train&Valid/MFFdatasets',
                 set_size=7874,
                 batchsize=8,
                 epochs=200,
                 lr=0.0002,
                 gamma=0.88,
                 scheduler_step=1,
                 lmd=0.,
                 patience=6,
                 load_balance_weight=0.01):  # MoE负载均衡损失权重
        # Hyper parameters
        self.DEVICE = GPUorCPU().DEVICE
        self.DATAPATH = data_path
        self.SETSIEZE = set_size
        self.BATCHSIZE = batchsize
        self.EPOCHS = epochs
        self.LR = lr
        self.GAMMA = gamma
        self.SCHEDULER_STEP = scheduler_step
        self.LMD = lmd
        self.PATIENCE = patience
        self.LOAD_BALANCE_WEIGHT = load_balance_weight
        # Form parameter dictionary
        self.hyperparas = {'set_size': self.SETSIEZE, 'batchsize': self.BATCHSIZE,
                           'epochs': self.EPOCHS, 'lr': self.LR, 'gamma': self.GAMMA,
                           'scheduler_step': self.SCHEDULER_STEP, 'lmd': self.LMD,
                           'patience': self.PATIENCE, 'load_balance_weight': self.LOAD_BALANCE_WEIGHT}

    def __call__(self, *args, **kwargs):
        TRAIN_LOADER, VALID_LOADER = self.PrepareDataLoader(self.DATAPATH, self.SETSIEZE, self.BATCHSIZE)
        MODEL, OPTIMIZER, SCHEDULER = self.BuildModel(self.DEVICE, self.LR, self.SCHEDULER_STEP, self.GAMMA)
        self.TrainingProcess(MODEL, OPTIMIZER, SCHEDULER, TRAIN_LOADER, VALID_LOADER, self.EPOCHS, self.LMD)

    def PrepareDataLoader(self, datapath, setsize, batchsize):
        train_list_A = sorted(glob.glob(os.path.join(datapath, 'train/sourceA', '*.*')))[:setsize]
        train_list_B = sorted(glob.glob(os.path.join(datapath, 'train/sourceB', '*.*')))[:setsize]
        train_list_GT = sorted(glob.glob(os.path.join(datapath, 'train/groundtruth', '*.*')))[:setsize]
        train_list_DM = sorted(glob.glob(os.path.join(datapath, 'train/decisionmap', '*.*')))[:setsize]
        valid_list_A = sorted(glob.glob(os.path.join(datapath, 'validate/sourceA', '*.*')))[:setsize // 9]
        valid_list_B = sorted(glob.glob(os.path.join(datapath, 'validate/sourceB', '*.*')))[:setsize // 9]
        valid_list_GT = sorted(glob.glob(os.path.join(datapath, 'validate/groundtruth', '*.*')))[:setsize // 9]
        valid_list_DM = sorted(glob.glob(os.path.join(datapath, 'validate/decisionmap', '*.*')))[:setsize // 9]
        tqdm.write(f"Train Data A: {len(train_list_A)}")
        tqdm.write(f"Train Data B: {len(train_list_B)}")
        tqdm.write(f"Train Data GT: {len(train_list_GT)}\n")
        tqdm.write(f"Valid Data A: {len(valid_list_A)}")
        tqdm.write(f"Valid Data B: {len(valid_list_B)}")
        tqdm.write(f"Valid Data GT: {len(valid_list_GT)}\n")
        train_data = DLr.DataLoader_Train(train_list_A, train_list_B, train_list_GT, train_list_DM)
        valid_data = DLr.DataLoader_Train(valid_list_A, valid_list_B, valid_list_GT, valid_list_DM)
        train_loader = DataLoader(dataset=train_data,
                                  batch_size=batchsize,
                                  shuffle=True,
                                  num_workers=0,
                                  pin_memory=False)
        valid_loader = DataLoader(dataset=valid_data,
                                  batch_size=batchsize,
                                  shuffle=True,
                                  num_workers=0,
                                  pin_memory=False)
        tqdm.write(f"Train Data Size:{len(train_data)} , Train Loader Amount: {len(train_data)}/{batchsize} = {len(train_loader)}")
        tqdm.write(f"Valid Data Size:{len(valid_data)} , Valid Loader Amount: {len(valid_data)}/{batchsize} = {len(valid_loader)}\n")
        return train_loader, valid_loader

    def BuildModel(self, device, lr, scheduler_step, gamma):
        model = Network().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, scheduler_step, gamma=gamma)
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print("The number of model parameters: {} M\n\n".format(round(num_params / 10e5, 6)))
        return model, optimizer, scheduler

    def MixLoss(self, pre, GT, lmd=0.):
        loss_abs = nn.L1Loss()  # loss_abs = torch.mean(torch.abs(GT - NetOut))
        loss = (1 - lmd) * loss_abs(GT, pre)
        # loss_mse = nn.MSELoss()
        # loss = 0.8*loss_l1(pre, GT) + 0.2*loss_mse(pre, GT)
        # Loss=HybridLoss()
        # loss = Loss(pre, GT)
        return loss
    
    def LoadBalancingLoss(self, gates_list):
        """
        负载均衡损失：鼓励专家均匀使用
        Args:
            gates_list: List of gate tensors, each shape [B, num_experts]
        Returns:
            balance_loss: 标量损失值
        """
        if gates_list is None or len(gates_list) == 0:
            return torch.tensor(0.0, device=self.DEVICE)
        
        total_balance_loss = 0.0
        for gates in gates_list:
            if gates is None:
                continue
            # gates: [B, num_experts] 或 List of [B, num_experts]
            if isinstance(gates, list):
                for gate in gates:
                    if gate is not None:
                        # 计算每个专家在batch中的平均使用率
                        expert_usage = gate.mean(dim=0)  # [num_experts]
                        # 计算使用率的方差（方差越小，越均衡）
                        balance_loss = torch.var(expert_usage)
                        total_balance_loss += balance_loss
            else:
                expert_usage = gates.mean(dim=0)  # [num_experts]
                balance_loss = torch.var(expert_usage)
                total_balance_loss += balance_loss
        
        # 平均所有层的负载均衡损失
        num_gates = sum(1 for g in gates_list if g is not None)
        if num_gates > 0:
            return total_balance_loss / num_gates
        return torch.tensor(0.0, device=self.DEVICE)

    def TrainingProcess(self, model, optimizer, scheduler, train_loader, valid_loader, epochs, lmd):
        scaler = torch.cuda.amp.GradScaler()
        torch.backends.cudnn.benchmark = True
        LS = Logging_SaveModel(savepath='RunTimeData', hyperparas=self.hyperparas)

        tqdm.write('Training start...\n')
        
        # MoE温度退火：初始化温度（训练初期温度较高，使分布更均匀）
        initial_temperature = 2.0
        final_temperature = 0.5
        # 在训练过程中动态调整温度
        def update_moe_temperature(model, epoch, total_epochs):
            """更新MoE门控网络的温度"""
            if total_epochs > 0:
                progress = epoch / total_epochs
                # 线性退火：从initial_temperature降到final_temperature
                current_temp = initial_temperature * (1 - progress) + final_temperature * progress
                # 更新所有AMoFE模块的温度
                for module in model.modules():
                    if hasattr(module, 'gate') and hasattr(module.gate, 'temperature'):
                        module.gate.temperature.fill_(current_temp)

        for epoch in range(epochs):
            # 更新MoE温度（每个epoch开始时）
            update_moe_temperature(model, epoch, epochs)
            
            ######################################### Train #########################################
            x = 0
            epoch_loss = 0
            epoch_accuracy = 0
            train_loader_tqdm = tqdm(train_loader, colour='green', leave=False, file=sys.stdout)
            
            for A, B, GT, DM in train_loader_tqdm:
                x += 1
                optimizer.zero_grad()
                
                # Automatic mixed precision training.
                with torch.autocast(device_type=self.DEVICE, dtype=torch.float16):
                    output = model(A, B)
                    
                    # 处理模型输出
                    if isinstance(output, tuple):
                        if len(output) == 3:
                            pre, gates_a, gates_b = output
                            # 合并两个分支的gates用于负载均衡计算
                            gates = [gates_a, gates_b] if (gates_a is not None or gates_b is not None) else None
                        else:
                            pre, gates = output
                    else:
                        pre = output
                        gates = None
                    
                    # 1. 计算主损失 (MixLoss: BCE + SSIM + Dice + TV)
                    loss = self.MixLoss(pre=pre, GT=DM)
                    
                    # 2. 添加MoE负载均衡损失
                    if gates is not None and self.LOAD_BALANCE_WEIGHT > 0:
                        balance_loss = self.LoadBalancingLoss(gates)
                        loss = loss + self.LOAD_BALANCE_WEIGHT * balance_loss
                
                # 反向传播与优化
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                # 3. 计算准确率 (Pixel Accuracy)
                # 二值化：>0.5 为前景(1)，否则为背景(0)
                pre_binary = (pre > 0.5).float()
                
                # 统计正确像素比例
                correct_pixels = (pre_binary == DM).sum().item()
                total_pixels = DM.numel()
                batch_acc = correct_pixels / total_pixels
                
                # 4. 累加指标
                epoch_loss += loss.item() / len(train_loader)
                epoch_accuracy += batch_acc / len(train_loader)
                
                # 5. 更新进度条显示
                train_loader_tqdm.set_description("[%s] Epoch %s" % (str(datetime.datetime.now().strftime('%Y-%m-%d %H.%M.%S')), str(epoch + 1)))
                train_loader_tqdm.set_postfix(loss=f"{loss.item():.4f}", acc=f"{batch_acc:.4f}")
                ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
            #########################################################################################
            ######################################### Valid #########################################
            with torch.no_grad():
                epoch_val_accuracy = 0
                epoch_val_loss = 0
                valid_loader_tqdm = tqdm(valid_loader, colour='yellow', leave=False, file=sys.stdout)
                val_x = 0
                
                for A, B, GT, DM in valid_loader_tqdm:
                    val_x += 1
                    output = model(A, B)
                    
                    # --- 1. 处理模型输出 ---
                    if isinstance(output, tuple):
                        if len(output) == 3:
                            pre, gates_a, gates_b = output
                            gates = [gates_a, gates_b] if (gates_a is not None or gates_b is not None) else None
                        else:
                            pre, gates = output
                    else:
                        pre = output
                        gates = None
                    
                    # --- 2. 计算 Loss ---
                    loss = self.MixLoss(pre=pre, GT=DM)
                    
                    # --- 3. 计算真正的准确率 (Pixel Accuracy) ---
                    # 将概率二值化：大于 0.5 为前景(1)，小于 0.5 为背景(0)
                    pre_binary = (pre > 0.5).float()
                    
                    # 计算正确的像素数 (Prediction == GroundTruth)
                    # DM 必须也是 0 或 1
                    correct_pixels = (pre_binary == DM).sum().item()
                    total_pixels = DM.numel() # 总像素数
                    
                    batch_acc = correct_pixels / total_pixels
                    
                    # --- 4. 累加指标 ---
                    # 注意：loss 还是张量，累加时建议转为 float 节省显存
                    epoch_val_accuracy += batch_acc / len(valid_loader)
                    epoch_val_loss += loss.item() / len(valid_loader)
                    
                    # --- 5. 显示进度 ---
                    valid_loader_tqdm.set_description("[Validating...] Epoch %s" % str(epoch + 1))
                    # 显示真实的 Loss 和 Acc
                    valid_loader_tqdm.set_postfix(loss=f"{loss.item():.4f}", acc=f"{batch_acc:.4f}")
                    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
            #########################################################################################
            # Print epoch loss and accuracy.
            tqdm.write(f"[{str(datetime.datetime.now().strftime('%Y-%m-%d %H.%M.%S'))}] Epoch {epoch + 1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}")
            # Dynamic learning rate.
            scheduler.step()
            # Logging and Save model weights.
            log_contents = f"Epoch {epoch + 1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
            LS(model, epoch + 1, log_contents, epoch_val_loss, save_every_model=True)
            if LS.ENDTRAIN:
                # Early stopping mechanism has been triggered.
                print("Early stopping!!!")
                # End training.
                break
        # Epoch loop ends.


if __name__ == '__main__':
    t = NetTrain()
    t()