import torch
import numpy as np
from skimage.segmentation import slic
#from Utilities.CUDA_Check import GPUorCPU
from torch_geometric.data import Data
import torch.nn.functional as F
class GPUorCPU:
    if torch.cuda.is_available():
        DEVICE = "cuda"
        # print('\nCUDA is available. Calculation is performing on ' + str(
        #     torch.cuda.get_device_name(torch.cuda.current_device())) + '.\n')
    else:
        DEVICE = 'cpu'
        # print('\nOOPS! CUDA is not available! Calculation is performing on CPU.\n')
DEVICE = GPUorCPU.DEVICE


def gradient(x):
    _, c, _, _ = x.shape
    device = x.device
    kernel = torch.tensor([[[[0, 1, 0],
                            [1, -4, 1],
                            [0, 1, 0]]]], dtype=x.dtype, device=device, requires_grad=False)
    kernel = kernel.repeat(1, c, 1, 1)
    grad = F.conv2d(x, kernel, padding=1)
    return grad


def sobel(x):
    _, c, _, _ = x.shape
    device = x.device
    kernel_x = torch.tensor([[[[-3, 0, 3],
                               [-10, 0, 10],
                               [-3, 0, 3]]]], dtype=x.dtype, device=device, requires_grad=False)
    kernel_y = torch.tensor([[[[3, 10, 3],
                               [0, 0, 0],
                               [-3, -10, -3]]]], dtype=x.dtype, device=device, requires_grad=False)
    
    kernel_x = kernel_x.repeat(c, 1, 1, 1)
    kernel_y = kernel_y.repeat(c, 1, 1, 1)
    
    edge_x = F.conv2d(x, kernel_x, padding=1, groups=c)
    edge_y = F.conv2d(x, kernel_y, padding=1, groups=c)
    edge = torch.sqrt(edge_x ** 2 + edge_y ** 2)
    return edge

def SegmentsLabelProcess(labels):
    labels = np.array(labels, np.int64)
    H, W = labels.shape
    ls = list(set(np.reshape(labels, [-1]).tolist()))
    dic = {}
    for i in range(len(ls)):
        dic[ls[i]] = i
    new_labels = labels
    for i in range(H):
        for j in range(W):
            new_labels[i, j] = dic[new_labels[i, j]]

    return new_labels


def SILC_Processes(img_a, img_b, fea_img, scale=0.5):

    _, c, h, w = img_a.shape
    fea_img = fea_img.clone().detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    init_segments = int(max(h, w) * scale)
    edge_list = []
    edge_img_a = sobel(img_a)
    edge_img_b = sobel(img_b)
    edge_fea_list_a = edge_img_a.clone().detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    edge_fea_list_b = edge_img_b.clone().detach().cpu().squeeze(0).permute(1, 2, 0).numpy()

    segments = slic(fea_img, n_segments=init_segments, sigma=5)
    if segments.max() + 1 != len(list(set(np.reshape(segments, [-1]).tolist()))): segments = SegmentsLabelProcess(segments)
    superpixel_num = segments.max() + 1  # slic划分出的超像素个数
    segments_flatten = np.reshape(segments, [-1])

    fea_matrix_a = np.zeros([superpixel_num, c], dtype=np.float32)  # 初始化gcn运算特征矩阵 [num, channels]
    fea_matrix_b = np.zeros([superpixel_num, c], dtype=np.float32)  # 初始化gcn运算特征矩阵 [num, channels]
    trans_matrix = torch.zeros([h*w, superpixel_num], dtype=torch.float32).to(DEVICE)  # 转换矩阵，将其特征提取后的转换到源图像上对应 [h*w, num]
    flatten_edge_img_a = np.reshape(edge_fea_list_a, [-1, c])
    flatten_edge_img_b = np.reshape(edge_fea_list_b, [-1, c])

    for i in range(superpixel_num):
        idx = np.where(segments_flatten == i)[0]
        edge_a = flatten_edge_img_a[idx]
        edge_fea_a = np.sum(edge_a, 0)
        edge_b = flatten_edge_img_b[idx]
        edge_fea_b = np.sum(edge_b, 0)

        fea_matrix_a[i] = edge_fea_a
        fea_matrix_b[i] = edge_fea_b
        trans_matrix[idx, i] = 1

    segments_ids = np.unique(segments)
    vs_right = np.vstack([segments[:, :-1].ravel(), segments[:, 1:].ravel()])
    vs_below = np.vstack([segments[:-1, :].ravel(), segments[1:, :].ravel()])
    bneighbors = np.unique(np.hstack([vs_right, vs_below]), axis=1)

    for i in range(bneighbors.shape[1]):
        node1 = bneighbors[0, i]
        node2 = bneighbors[1, i]
        idx1 = np.where(segments_ids == node1)[0][0]
        idx2 = np.where(segments_ids == node2)[0][0]
        edge_list.append(idx1)
        edge_list.append(idx2)

    # Add self loops
    for i in range(len(segments_ids)):
        edge_list.append(i)
        edge_list.append(i)

    fea_matrix_a = torch.from_numpy(fea_matrix_a).to(DEVICE)
    fea_matrix_b = torch.from_numpy(fea_matrix_b).to(DEVICE)

    edge_index = torch.tensor(edge_list).view(-1, 2).to(DEVICE)

    data_a = Data(x=fea_matrix_a, edge_index=edge_index.t())
    data_b = Data(x=fea_matrix_b, edge_index=edge_index.t())
    return data_a, data_b, trans_matrix


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data


# 假设 DEVICE 已经在外部定义，或者你可以取消下面的注释
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def apply_laplacian_enhancement(features_flat, spatial_shape, laplacian_weight=0.3):
    """
    向量化优化的拉普拉斯增强
    """
    H, W = spatial_shape
    N, C = features_flat.shape

    # [N, C] -> [1, C, H, W]
    features_spatial = features_flat.view(H, W, C).permute(2, 0, 1).unsqueeze(0)

    # 定义拉普拉斯核 (固定不可变，不需要梯度)
    laplacian_kernel_base = torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]],
                                         dtype=features_spatial.dtype, device=features_spatial.device)
    laplacian_kernel = laplacian_kernel_base.repeat(C, 1, 1, 1)

    # 分组卷积
    with torch.no_grad():  # 通常拉普拉斯边缘不需要反向传播，节省显存
        laplacian_features = F.conv2d(features_spatial, laplacian_kernel, padding=1, groups=C)
        laplacian_features = torch.abs(laplacian_features)

    laplacian_flat = laplacian_features.squeeze(0).permute(1, 2, 0).reshape(N, C)

    laplacian_norm = F.normalize(laplacian_flat, p=2, dim=1)
    features_norm = F.normalize(features_flat, p=2, dim=1)

    enhanced_features = (1 - laplacian_weight) * features_norm + laplacian_weight * laplacian_norm
    return enhanced_features


def build_grid_trans_matrix(h, w, h_s, w_s, stride, device):
    """
    构建规则网格的转换矩阵（稀疏矩阵）
    
    Args:
        h, w: 原图尺寸
        h_s, w_s: 下采样后尺寸 (h_s = h // stride, w_s = w // stride)
        stride: 下采样步长
        device: 设备
    
    Returns:
        trans_matrix: [h*w, h_s*w_s] 稀疏矩阵
    """
    N_large = h * w
    N_small = h_s * w_s
    
    indices_list = []
    values_list = []
    
    for i in range(h_s):
        for j in range(w_s):
            small_idx = i * w_s + j
            
            i_start = i * stride
            j_start = j * stride
            i_end = min((i + 1) * stride, h)
            j_end = min((j + 1) * stride, w)
            
            area = (i_end - i_start) * (j_end - j_start)
            weight = 1.0 / area
            
            for ii in range(i_start, i_end):
                for jj in range(j_start, j_end):
                    large_idx = ii * w + jj
                    indices_list.append([large_idx, small_idx])
                    values_list.append(weight)
    
    indices = torch.tensor(indices_list, dtype=torch.long, device=device).t()
    values = torch.tensor(values_list, dtype=torch.float32, device=device)
    
    trans_matrix = torch.sparse_coo_tensor(
        indices, values, (N_large, N_small), device=device
    )
    
    return trans_matrix


def optimized_knn_graph_construction(features, positions, k, spatial_radius):
    """
    高度向量化的 k-NN 图构建，融合了相似度计算、空间约束和构图
    """
    N = features.shape[0]
    device = features.device

    features_norm = F.normalize(features, p=2, dim=1)
    sim_matrix = torch.mm(features_norm, features_norm.t())

    if positions is not None:
        spatial_dist = torch.cdist(positions.float(), positions.float(), p=2)
        mask = spatial_dist > spatial_radius
        sim_matrix.masked_fill_(mask, float('-inf'))

    sim_matrix.fill_diagonal_(float('-inf'))

    k_actual = min(k, N - 1)
    _, topk_indices = torch.topk(sim_matrix, k=k_actual, dim=1)

    source_nodes = torch.arange(N, device=device).repeat_interleave(k_actual)
    target_nodes = topk_indices.view(-1)

    valid_mask = sim_matrix[source_nodes, target_nodes] > float('-inf')
    source_nodes = source_nodes[valid_mask]
    target_nodes = target_nodes[valid_mask]

    edges = torch.stack([source_nodes, target_nodes], dim=0)
    edges_rev = torch.stack([target_nodes, source_nodes], dim=0)
    edge_index = torch.cat([edges, edges_rev], dim=1)

    self_loops = torch.arange(N, device=device).repeat(2, 1)
    edge_index = torch.cat([edge_index, self_loops], dim=1)

    if device.type == 'mps':
        edge_index = torch.unique(edge_index.cpu(), dim=1).to(device)
    else:
        edge_index = torch.unique(edge_index, dim=1)

    return edge_index


def FeatureSimilarityGraph(img_a, img_b, fea_img, k=8, spatial_radius=5.0,
                           similarity_metric='cosine', use_edge_features=True,
                           grid_stride=1, use_laplacian=False, laplacian_weight=0.3,
                           return_trans_matrix=True):
    b, c, h, w = img_a.shape
    DEVICE = img_a.device

    if grid_stride > 1:
        h_s, w_s = h // grid_stride, w // grid_stride
        fea_a_small = F.interpolate(img_a, size=(h_s, w_s), mode='bilinear', align_corners=True)
        fea_b_small = F.interpolate(img_b, size=(h_s, w_s), mode='bilinear', align_corners=True)
        fea_img_small = F.interpolate(fea_img, size=(h_s, w_s), mode='bilinear', align_corners=True)
    else:
        h_s, w_s = h, w
        fea_a_small = img_a
        fea_b_small = img_b
        fea_img_small = fea_img

    if use_edge_features:
        edge_a = sobel(fea_a_small).permute(0, 2, 3, 1)
        fea_a_ready = edge_a.reshape(-1, edge_a.shape[-1])

        edge_b = sobel(fea_b_small).permute(0, 2, 3, 1)
        fea_b_ready = edge_b.reshape(-1, edge_b.shape[-1])

        edge_metric = sobel(fea_img_small).permute(0, 2, 3, 1)
        fea_metric = edge_metric.reshape(-1, edge_metric.shape[-1])
    else:
        fea_a_ready = fea_a_small.permute(0, 2, 3, 1).reshape(-1, c)
        fea_b_ready = fea_b_small.permute(0, 2, 3, 1).reshape(-1, c)
        fea_metric = fea_img_small.permute(0, 2, 3, 1).reshape(-1, c)

    N = h_s * w_s

    if use_laplacian:
        fea_metric = apply_laplacian_enhancement(fea_metric, (h_s, w_s), laplacian_weight)

    y_coords, x_coords = torch.meshgrid(
        torch.arange(h_s, device=DEVICE, dtype=torch.float32),
        torch.arange(w_s, device=DEVICE, dtype=torch.float32),
        indexing='ij'
    )
    node_positions = torch.stack([y_coords.ravel(), x_coords.ravel()], dim=1)
    effective_radius = spatial_radius / grid_stride

    edge_index = optimized_knn_graph_construction(
        fea_metric,
        node_positions,
        k=k,
        spatial_radius=effective_radius
    )

    data_a = Data(x=fea_a_ready, edge_index=edge_index)
    data_b = Data(x=fea_b_ready, edge_index=edge_index)

    if return_trans_matrix and grid_stride > 1:
        trans_matrix = build_grid_trans_matrix(h, w, h_s, w_s, grid_stride, DEVICE)
    else:
        trans_matrix = None

    return data_a, data_b, trans_matrix
