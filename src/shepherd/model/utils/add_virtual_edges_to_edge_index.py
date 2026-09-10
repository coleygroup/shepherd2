import torch

def add_virtual_edges_to_edge_index(edge_index, virtual_node_mask, batch):
    """
    Adds edges to edge_index that connect all (real) nodes to the virtual node(s)

    Arguments:
        edge_index -- torch.LongTensor with shape (2, N_edges)
        virtual_node_mask -- torch.BoolTensor with shape (N_nodes,)
        batch -- torch.LongTensor with shape (N_nodes,)

    Returns:
        new_edge_index -- updated edge_index with additional virtual edges
    """
    # edge_index (2, N_edges)
    # virtual_node_mask (N_nodes,) -- boolean tensor where True indicates a virtual node
    # batch (N_nodes,)

    # remove existing edges to/from virtual nodes, to avoid duplicating edges
    edge_mask = virtual_node_mask[edge_index[1]] | virtual_node_mask[edge_index[0]]
    edge_index_without_VN = edge_index[:, ~edge_mask]

    # This previously called radius_graph(zeros, r=np.inf, batch, max_num_neighbors=1e6),
    idx = torch.arange(batch.shape[0], device = batch.device)
    real_idx = idx[~virtual_node_mask]
    virt_idx = idx[virtual_node_mask]

    same_molecule = batch[real_idx].unsqueeze(1) == batch[virt_idx].unsqueeze(0)
    r_sel, v_sel = torch.nonzero(same_molecule, as_tuple = True)
    r_nodes, v_nodes = real_idx[r_sel], virt_idx[v_sel]
    src = [r_nodes, v_nodes]
    dst = [v_nodes, r_nodes]

    if virt_idx.numel() > 1:
        virt_same_molecule = batch[virt_idx].unsqueeze(1) == batch[virt_idx].unsqueeze(0)
        virt_same_molecule.fill_diagonal_(False) # excludes self-loops, as radius_graph did
        a_sel, b_sel = torch.nonzero(virt_same_molecule, as_tuple = True)
        src.append(virt_idx[a_sel])
        dst.append(virt_idx[b_sel])

    edge_index_VN = torch.stack([torch.cat(src), torch.cat(dst)], dim = 0)

    # combine and return
    new_edge_index = torch.cat([edge_index_without_VN, edge_index_VN], dim = 1)
    return new_edge_index


def bond_weighting(input_tensor: torch.Tensor, graph_batch_index: torch.Tensor, virtual_node_x1: bool):
    """
    This redistributes the input tensor to match the bonds in the graph.

    Args:
        input_tensor: torch.Tensor of shape (N_bonds,)
        graph_batch_index: torch.Tensor of shape (N_nodes,) (torch geometric graph.batch)
        virtual_node_x1: bool (True if virtual node is present in the graph)

    Returns:
        torch.Tensor of shape (N_bonds,)
    """
    # torch geometric graph.batch -> e.g. [0, 0, 1, 1, 1]
    # This produces: [2, 3]
    batch_num_nodes = torch.bincount(graph_batch_index)
    # virtual node is not counted in edges - upper triangular fully connected graph
    num_real_atoms = batch_num_nodes - int(virtual_node_x1)
    num_bonds = num_real_atoms * (num_real_atoms - 1) / 2

    # if input_tensor = [1, 2], this produces: [1, 1, 2, 2, 2]
    input_tensor_bonds = torch.repeat_interleave(input_tensor, num_bonds.long())
    return input_tensor_bonds
