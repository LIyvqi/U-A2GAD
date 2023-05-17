from dgl.transforms.functional import *
import numpy as np

from dgl.base import dgl_warning, DGLError, NID, EID
from dgl import convert
from dgl import backend as F
from dgl.sampling.neighbor import sample_neighbors
from dgl.transforms.functional import knn
from dgl.transforms.functional import pairwise_squared_distance

def _knn_graph_blas(x, k, dist='euclidean'):
    if F.ndim(x) == 2:
        x = F.unsqueeze(x, 0)
    n_samples, n_points, _ = F.shape(x)

    if k > n_points:
        dgl_warning("'k' should be less than or equal to the number of points in 'x'" \
                    "expect k <= {0}, got k = {1}, use k = {0}".format(n_points, k))
        k = n_points

    # if use cosine distance, normalize input points first
    # thus we can use euclidean distance to find knn equivalently.
    if dist == 'cosine':
        l2_norm = lambda v: F.sqrt(F.sum(v * v, dim=2, keepdims=True))
        x = x / (l2_norm(x) + 1e-5)

    ctx = F.context(x)
    dist = pairwise_squared_distance(x)
    k_indices = F.astype(F.argtopk(dist, k, 2, descending=True), F.int64)
    # index offset for each sample
    offset = F.arange(0, n_samples, ctx=ctx) * n_points
    offset = F.unsqueeze(offset, 1)
    src = F.reshape(k_indices, (n_samples, n_points * k))
    src = F.unsqueeze(src, 0) + offset
    dst = F.repeat(F.arange(0, n_points, ctx=ctx), k, dim=0)
    dst = F.unsqueeze(dst, 0) + offset
    return convert.graph((F.reshape(src, (-1,)), F.reshape(dst, (-1,))))

def neg_knn_graph(x, k, algorithm='bruteforce-blas', dist='euclidean',
              exclude_self=False):

    if exclude_self:
        # add 1 to k, for the self edge, since it will be removed
        k = k + 1

    # check invalid k
    if k <= 0:
        raise DGLError("Invalid k value. expect k > 0, got k = {}".format(k))

    # check empty point set
    x_size = tuple(F.shape(x))
    if x_size[0] == 0:
        raise DGLError("Find empty point set")

    d = F.ndim(x)
    x_seg = x_size[0] * [x_size[1]] if d == 3 else [x_size[0]]
    if algorithm == 'bruteforce-blas':
        result = _knn_graph_blas(x, k, dist=dist)
    else:
        if d == 3:
            x = F.reshape(x, (x_size[0] * x_size[1], x_size[2]))
        out = knn(k, x, x_seg, algorithm=algorithm, dist=dist)
        row, col = out[1], out[0]
        result = convert.graph((row, col))

    if d == 3:
        # set batch information if x is 3D
        num_nodes = F.tensor(x_seg, dtype=F.int64).to(F.context(x))
        result.set_batch_num_nodes(num_nodes)
        # if any segment is too small for k, all algorithms reduce k for all segments
        clamped_k = min(k, np.min(x_seg))
        result.set_batch_num_edges(clamped_k*num_nodes)

    if exclude_self:
        # remove_self_loop will update batch_num_edges as needed
        result = remove_self_loop(result)

        # If there were more than k(+1) coincident points, there may not have been self loops on
        # all nodes, in which case there would still be one too many out edges on some nodes.
        # However, if every node had a self edge, the common case, every node would still have the
        # same degree as each other, so we can check that condition easily.
        # The -1 is for the self edge removal.
        clamped_k = min(k, np.min(x_seg)) - 1
        if result.num_edges() != clamped_k*result.num_nodes():
            # edges on any nodes with too high degree should all be length zero,
            # so pick an arbitrary one to remove from each such node
            degrees = result.in_degrees()
            node_indices = F.nonzero_1d(degrees > clamped_k)
            edges_to_remove_graph = sample_neighbors(result, node_indices, 1, edge_dir='in')
            edge_ids = edges_to_remove_graph.edata[EID]
            result = remove_edges(result, edge_ids)

    return result


# test
# import torch
# import dgl
# X_fea = torch.tensor([[1],[2],[3],[4],[5]])
# g = neg_knn_graph(X_fea,1,exclude_self=True)
# g = dgl.to_bidirected(g)
# print(g.edges())
