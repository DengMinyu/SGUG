import torch
import torch.nn.functional as F
from torch_geometric.data import Data


# -------------------------------------------------
# 1. 聚焦不确定性（为多聚焦优化）
# -------------------------------------------------
def get_focus_uncertainty(features):
    """
    features: [B, C, H, W]
    return: [B, 1, H, W]
    """
    energy = torch.sqrt(torch.mean(features.float() ** 2, dim=1, keepdim=True))

    lap_kernel = torch.tensor(
        [[[[0, 1, 0],
           [1, -4, 1],
           [0, 1, 0]]]],
        device=features.device,
        dtype=energy.dtype
    )

    edge = torch.abs(F.conv2d(energy, lap_kernel, padding=1))

    B = edge.shape[0]
    flat = edge.view(B, -1)
    min_v = flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    max_v = flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)

    return (edge - min_v) / (max_v - min_v + 1e-6)


# -------------------------------------------------
# 2. 自适应节点采样 + soft 转换矩阵（修正版）
# -------------------------------------------------
def adaptive_node_sampling(
    features,
    easy_stride=4,
    hard_quantile=0.7,
    sigma=1.5
):
    B, C, H, W = features.shape
    device = features.device

    uncertainty = get_focus_uncertainty(features)

    y_grid, x_grid = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )

    node_list, coord_list, batch_list = [], [], []
    pixel_idx_all, node_idx_all, value_all = [], [], []

    global_node_offset = 0
    global_pixel_offset = 0

    for b in range(B):
        u = uncertainty[b, 0]

        # ✅ 修复 1：quantile dtype
        u_q = u.float() if not u.is_floating_point() else u
        thr = torch.quantile(u_q.view(-1), hard_quantile)

        grid_mask = torch.zeros_like(u, dtype=torch.bool)
        grid_mask[::easy_stride, ::easy_stride] = True
        hard_mask = u > thr
        final_mask = grid_mask | hard_mask

        num_nodes = int(final_mask.sum())
        if num_nodes == 0:
            continue

        feat_b = features[b].permute(1, 2, 0)
        nodes = feat_b[final_mask]
        coords = torch.stack(
            [y_grid[final_mask], x_grid[final_mask]], dim=1
        )

        node_list.append(nodes)
        coord_list.append(coords)
        batch_list.append(
            torch.full((num_nodes,), b, device=device, dtype=torch.long)
        )

        node_id_map = torch.full((H, W), -1, device=device, dtype=torch.long)
        node_id_map[final_mask] = torch.arange(num_nodes, device=device)

        # parent grid
        py = (y_grid // easy_stride) * easy_stride
        px = (x_grid // easy_stride) * easy_stride
        py.clamp_(0, H - 1)
        px.clamp_(0, W - 1)

        parent_id = node_id_map[py, px]

        # ✅ 修复 2：确保 mapping 中没有 -1
        mapping = torch.where(
            node_id_map >= 0,
            node_id_map,
            parent_id
        )
        mapping = torch.clamp(mapping, min=0)

        node_ids = mapping.view(-1)

        node_y = coords[:, 0]
        node_x = coords[:, 1]

        pixel_y = y_grid.view(-1)
        pixel_x = x_grid.view(-1)

        dy = pixel_y - node_y[node_ids]
        dx = pixel_x - node_x[node_ids]
        dist2 = dy.float() ** 2 + dx.float() ** 2

        weight = torch.exp(-dist2 / (2 * sigma ** 2))

        # ✅ 修复 3：pixel-wise 归一化（非常重要）
        weight = weight / (weight.mean() + 1e-6)

        pixel_indices = torch.arange(
            global_pixel_offset,
            global_pixel_offset + H * W,
            device=device
        )
        node_indices = node_ids + global_node_offset

        pixel_idx_all.append(pixel_indices)
        node_idx_all.append(node_indices)
        value_all.append(weight)

        global_node_offset += num_nodes
        global_pixel_offset += H * W

    if len(node_list) == 0:
        return None, None, None, None

    all_nodes = torch.cat(node_list, dim=0)
    all_coords = torch.cat(coord_list, dim=0)
    batch_vector = torch.cat(batch_list, dim=0)

    indices = torch.stack(
        [torch.cat(pixel_idx_all), torch.cat(node_idx_all)], dim=0
    )
    values = torch.cat(value_all)
    shape = (B * H * W, all_nodes.shape[0])

    return all_nodes, all_coords, batch_vector, (indices, values, shape)


# -------------------------------------------------
# 3. KNN 图（保持你原来的，仅安全微调）
# -------------------------------------------------
def optimized_knn_graph_construction(
    features, positions, batch_vector, k=8, spatial_radius=5.0
):
    device = features.device
    edge_list = []

    for b in torch.unique(batch_vector):
        mask = batch_vector == b
        idx = torch.nonzero(mask).squeeze(1)
        if idx.numel() <= 1:
            continue

        f = F.normalize(features[mask].float(), dim=1)
        p = positions[mask].float()

        sim = torch.mm(f, f.t())
        dist = torch.cdist(p, p)
        sim[dist > spatial_radius] = -1e9
        sim.fill_diagonal_(-1e9)

        k_eff = min(k, sim.shape[0] - 1)
        _, nbr = torch.topk(sim, k_eff, dim=1)

        src = torch.arange(sim.shape[0], device=device).repeat_interleave(k_eff)
        dst = nbr.reshape(-1)

        edge_list.append(torch.stack([idx[src], idx[dst]], dim=0))
        edge_list.append(torch.stack([idx[dst], idx[src]], dim=0))

    if len(edge_list) == 0:
        edge_index = torch.empty((2, 0), device=device, dtype=torch.long)
    else:
        edge_index = torch.cat(edge_list, dim=1)

    self_loop = torch.arange(features.shape[0], device=device)
    edge_index = torch.cat(
        [edge_index, torch.stack([self_loop, self_loop])], dim=1
    )

    return torch.unique(edge_index, dim=1)


# -------------------------------------------------
# 4. 主接口（不改调用方式）
# -------------------------------------------------
def FeatureSimilarityGraph_Adaptive(
    img_a, img_b, fea_img,
    k=12,
    easy_stride=4,
    hard_quantile=0.7,
    return_trans_matrix=True
):
    device = img_a.device
    radius = easy_stride * 2

    nodes, coords, batch, trans_info = adaptive_node_sampling(
        fea_img,
        easy_stride=easy_stride,
        hard_quantile=hard_quantile
    )

    if nodes is None:
        return None, None, None

    y, x = coords[:, 0], coords[:, 1]

    fea_a = img_a[batch, :, y, x]
    fea_b = img_b[batch, :, y, x]

    edge_index = optimized_knn_graph_construction(
        nodes, coords, batch, k, radius
    )

    data_a = Data(x=fea_a, edge_index=edge_index, batch=batch)
    data_b = Data(x=fea_b, edge_index=edge_index, batch=batch)

    trans_matrix = None
    if return_trans_matrix:
        i, v, s = trans_info
        trans_matrix = torch.sparse_coo_tensor(i, v, s, device=device)

    return data_a, data_b, trans_matrix
