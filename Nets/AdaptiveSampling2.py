import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

def get_laplacian_uncertainty(features):
    """
    计算特征图的拉普拉斯不确定性
    Returns: [B, 1, H, W]
    """
    if features.shape[1] > 1:
        feat_intensity = torch.mean(features, dim=1, keepdim=True)
    else:
        feat_intensity = features

    kernel = torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], 
                          dtype=features.dtype, device=features.device)
    
    edge_map = F.conv2d(feat_intensity, kernel, padding=1)
    edge_map = torch.abs(edge_map)
    
    B = edge_map.shape[0]
    flat_map = edge_map.view(B, -1)
    min_v = flat_map.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    max_v = flat_map.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    
    return (edge_map - min_v) / (max_v - min_v + 1e-6)


def adaptive_node_sampling(features, easy_stride=4, hard_threshold=0.3):
    """
    自适应采样逻辑
    Returns: 
        all_nodes: [N_total, C]
        all_coords: [N_total, 2]
        batch_vector: [N_total]
        trans_info: tuple for sparse matrix
    """
    B, C, H, W = features.shape
    device = features.device
    
    uncertainty_map = get_laplacian_uncertainty(features)
    
    node_list = []
    coord_list = []
    batch_idx_list = []
    
    pixel_indices_list = []
    node_indices_list = []
    global_node_offset = 0
    global_pixel_offset = 0
    
    y_grid, x_grid = torch.meshgrid(torch.arange(H, device=device), 
                                    torch.arange(W, device=device), indexing='ij')
    
    for b in range(B):
        u_map = uncertainty_map[b, 0]
        
        # Mask Logic
        grid_mask = torch.zeros((H, W), dtype=torch.bool, device=device)
        grid_mask[::easy_stride, ::easy_stride] = True
        hard_mask = u_map > hard_threshold
        final_mask = hard_mask | grid_mask
        
        num_nodes_b = final_mask.sum().item()
        
        # Extract features/coords
        feat_b = features[b].permute(1, 2, 0)
        active_nodes = feat_b[final_mask]
        active_coords = torch.stack([y_grid[final_mask], x_grid[final_mask]], dim=1)
        batch_ids = torch.full((num_nodes_b,), b, dtype=torch.long, device=device)
        
        node_list.append(active_nodes)
        coord_list.append(active_coords)
        batch_idx_list.append(batch_ids)
        
        # Build Mapping
        node_id_map = torch.full((H, W), -1, dtype=torch.long, device=device)
        node_id_map[final_mask] = torch.arange(num_nodes_b, device=device)
        
        parent_y = torch.clamp((y_grid // easy_stride) * easy_stride, 0, H-1)
        parent_x = torch.clamp((x_grid // easy_stride) * easy_stride, 0, W-1)
        parent_ids = node_id_map[parent_y, parent_x]
        
        mapping_local = torch.where(node_id_map >= 0, node_id_map, parent_ids)
        
        current_pixel_indices = torch.arange(global_pixel_offset, global_pixel_offset + H*W, device=device)
        current_node_indices = mapping_local.view(-1) + global_node_offset
        
        pixel_indices_list.append(current_pixel_indices)
        node_indices_list.append(current_node_indices)
        
        global_node_offset += num_nodes_b
        global_pixel_offset += H * W

    if len(node_list) > 0:
        all_nodes = torch.cat(node_list, dim=0)
        all_coords = torch.cat(coord_list, dim=0)
        batch_vector = torch.cat(batch_idx_list, dim=0)
        
        all_pixel_indices = torch.cat(pixel_indices_list, dim=0)
        all_node_mapping = torch.cat(node_indices_list, dim=0)
        trans_indices = torch.stack([all_pixel_indices, all_node_mapping], dim=0)
        trans_values = torch.ones(all_pixel_indices.shape[0], device=device)
        trans_shape = (B * H * W, all_nodes.shape[0])
        trans_info = (trans_indices, trans_values, trans_shape)
        
        return all_nodes, all_coords, batch_vector, trans_info
    else:
        return None, None, None, None


# def optimized_knn_graph_construction(features, positions, batch_vector, k, spatial_radius):
#     """
#     【关键修改】：增加 batch_vector 参数，并实现跨 Batch 屏蔽
#     """
#     N = features.shape[0]
#     device = features.device

#     features_norm = F.normalize(features, p=2, dim=1)
    
#     # 1. 计算相似度矩阵
#     sim_matrix = torch.mm(features_norm, features_norm.t())

#     # 2. 空间距离约束
#     if positions is not None:
#         spatial_dist = torch.cdist(positions.float(), positions.float(), p=2)
#         mask_spatial = spatial_dist > spatial_radius
#         sim_matrix.masked_fill_(mask_spatial, float('-inf'))

#     # 3. 【新增】Batch 隔离约束
#     # 逻辑：如果 node_i 和 node_j 不在同一个 batch，则相似度设为 -inf
#     if batch_vector is not None:
#         # batch_vector: [N]
#         # batch_row: [N, 1], batch_col: [1, N]
#         batch_row = batch_vector.unsqueeze(1)
#         batch_col = batch_vector.unsqueeze(0)
#         # 广播比较：如果不相等，说明不在同图，mask = True
#         mask_batch = batch_row != batch_col
#         sim_matrix.masked_fill_(mask_batch, float('-inf'))

#     # 4. 排除自环
#     sim_matrix.fill_diagonal_(float('-inf'))

#     # 5. Top-K 选择
#     k_actual = min(k, N - 1)
#     _, topk_indices = torch.topk(sim_matrix, k=k_actual, dim=1)

#     source_nodes = torch.arange(N, device=device).repeat_interleave(k_actual)
#     target_nodes = topk_indices.view(-1)

#     # 6. 过滤无效边
#     valid_mask = sim_matrix[source_nodes, target_nodes] > float('-inf')
#     source_nodes = source_nodes[valid_mask]
#     target_nodes = target_nodes[valid_mask]

#     edges = torch.stack([source_nodes, target_nodes], dim=0)
#     edges_rev = torch.stack([target_nodes, source_nodes], dim=0)
#     edge_index = torch.cat([edges, edges_rev], dim=1)
    
#     # 添加自环
#     self_loops = torch.arange(N, device=device).repeat(2, 1)
#     edge_index = torch.cat([edge_index, self_loops], dim=1)

#     if device.type == 'mps':
#         edge_index = torch.unique(edge_index.cpu(), dim=1).to(device)
#     else:
#         edge_index = torch.unique(edge_index, dim=1)

#     return edge_index
def optimized_knn_graph_construction(features, positions, batch_vector, k, spatial_radius):
    """
    内存优化版：逐图构建 KNN，避免生成巨大的全 Batch 相似度矩阵
    """
    device = features.device
    Total_Nodes = features.shape[0]
    
    # 容器用于收集所有边
    all_edges_list = []
    
    # 获取 batch 中包含的图片 ID (通常是 0, 1, 2... B-1)
    unique_batch_ids = torch.unique(batch_vector)
    
    # --- 关键修改：逐图循环 (Loop over images) ---
    # Python 循环开销在这里可以忽略，因为内部矩阵运算才是大头
    # 这样可以确保显存峰值只取决于“单张图的节点数”，而不是 Batch 大小
    for b_id in unique_batch_ids:
        # 1. 找到属于当前图片的节点掩码
        mask_b = (batch_vector == b_id)
        
        # 获取这些节点在全局列表中的索引 (用于最后映射回去)
        global_indices = torch.nonzero(mask_b).squeeze() # [N_b]
        if global_indices.dim() == 0: # 处理只有1个节点的情况
            global_indices = global_indices.unsqueeze(0)
            
        # 2. 提取当前图的特征和坐标
        sub_feat = features[mask_b]    # [N_b, C]
        sub_pos = positions[mask_b]    # [N_b, 2]
        n_b = sub_feat.shape[0]
        
        if n_b <= 1: # 节点太少无法构图，跳过
            continue

        # 3. 计算单图相似度矩阵 [N_b, N_b] (内存占用极小)
        sub_feat_norm = F.normalize(sub_feat, p=2, dim=1)
        sim_matrix = torch.mm(sub_feat_norm, sub_feat_norm.t())
        
        # 空间距离约束
        if sub_pos is not None:
            spatial_dist = torch.cdist(sub_pos.float(), sub_pos.float(), p=2)
            sim_matrix.masked_fill_(spatial_dist > spatial_radius, float('-inf'))
            
        # 排除自环
        sim_matrix.fill_diagonal_(float('-inf'))
        
        # 4. Top-K 选择
        k_actual = min(k, n_b - 1)
        _, topk_idx = torch.topk(sim_matrix, k=k_actual, dim=1) # [N_b, k]
        
        # 5. 构建边 (局部索引)
        # source_local: 0, 0, 0, 1, 1, 1 ...
        source_local = torch.arange(n_b, device=device).repeat_interleave(k_actual)
        target_local = topk_idx.view(-1)
        
        # 过滤无效边
        valid_mask = sim_matrix[source_local, target_local] > float('-inf')
        source_local = source_local[valid_mask]
        target_local = target_local[valid_mask]
        
        # 6. 映射回全局索引 (关键步骤)
        # 将局部索引 (0~N_b) 转换为全局索引 (0~Total_Nodes)
        source_global = global_indices[source_local]
        target_global = global_indices[target_local]
        
        # 7. 存入列表 (双向边)
        edges = torch.stack([source_global, target_global], dim=0)
        edges_rev = torch.stack([target_global, source_global], dim=0)
        
        all_edges_list.append(edges)
        all_edges_list.append(edges_rev)

    # --- 合并结果 ---
    if len(all_edges_list) > 0:
        edge_index = torch.cat(all_edges_list, dim=1)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

    # 添加全局自环 (Self-loops)
    self_loops = torch.arange(Total_Nodes, device=device).repeat(2, 1)
    edge_index = torch.cat([edge_index, self_loops], dim=1)
    
    # 去重
    if device.type == 'mps':
        edge_index = torch.unique(edge_index.cpu(), dim=1).to(device)
    else:
        edge_index = torch.unique(edge_index, dim=1)

    return edge_index

def FeatureSimilarityGraph_Adaptive(img_a, img_b, fea_img, k=12,
                                    easy_stride=4, hard_threshold=0.3,
                                    return_trans_matrix=True):
    """
    主函数
    """
    DEVICE = img_a.device
    radius = easy_stride * 2
    # 1. 自适应采样
    fea_nodes_metric, coords, batch_vector, trans_info = adaptive_node_sampling(
        fea_img, easy_stride=easy_stride, hard_threshold=hard_threshold
    )
    
    if fea_nodes_metric is None:
        return None, None, None
        
    # 2. 【高效提取】利用 Advanced Indexing 直接从原图提取
    b_idx = batch_vector
    y_idx = coords[:, 0]
    x_idx = coords[:, 1]
    
    # [B, C, H, W] -> index by [N], [N], [N] -> [N, C]
    fea_a_nodes = img_a[b_idx, :, y_idx, x_idx]
    fea_b_nodes = img_b[b_idx, :, y_idx, x_idx]
    
    # 3. 构建 KNN 图 (传入 batch_vector)
    edge_index = optimized_knn_graph_construction(
        fea_nodes_metric, 
        coords, 
        batch_vector,  # 传入 batch_vector
        k=k, 
        spatial_radius=radius
    )

    # 4. 封装 Data
    data_a = Data(x=fea_a_nodes, edge_index=edge_index, batch=batch_vector)
    data_b = Data(x=fea_b_nodes, edge_index=edge_index, batch=batch_vector)

    # 5. 稀疏矩阵
    trans_matrix = None
    if return_trans_matrix and trans_info is not None:
        indices, values, shape = trans_info
        trans_matrix = torch.sparse_coo_tensor(indices, values, shape, device=DEVICE)

    return data_a, data_b, trans_matrix