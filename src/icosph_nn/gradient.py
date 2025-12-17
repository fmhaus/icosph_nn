import torch.nn as nn
import torch
from icosph_nn import utils
from icosph_nn.icosphere import IcoSphere, cart2sph
from collections import defaultdict

def _create_weights(vertices, base_dirs, weight_grouping, indices, neighbour_indices):
    results = []
    for dir_group, base_vec in zip(weight_grouping, base_dirs):
        if len(dir_group) == 2:
            to_n1 = vertices[neighbour_indices[dir_group[0]]] - vertices[indices]
            to_n2 = vertices[neighbour_indices[dir_group[1]]] - vertices[indices]
            dot1 = torch.sum(to_n1 * base_vec[indices], dim=-1)
            dot2 = torch.sum(to_n2 * base_vec[indices], dim=-1)
            length = dot1**2 + dot2**2
            results.append(dot1 / length)
            results.append(dot2 / length)
        elif len(dir_group) == 1:
            to_n = vertices[dir_group[0]] - vertices[indices]
            dot = torch.sum(to_n * base_vec[indices], dim=-1)
            results.append(1 / dot)
        else:
            assert(False)
    
    return torch.stack(results, dim=0)

class NeighbourWeightTable:
    def __init__(self, L, device):
        HS, S = utils.get_face_side_dims(L)

        icosphere = IcoSphere(L, device)
        v_cart = icosphere.generate_vertices()
        V, _ = v_cart.shape
        
        N = (V-2) // 5
        v_sph = cart2sph(v_cart[:N])    

        north = IcoSphere.get_north_directions(v_sph)
        east = IcoSphere.get_east_directions(v_sph)
        base_dirs = (east, -east, north, -north)

        indices = torch.arange(N, device=device).view(HS, S)
        indices_next = indices + (HS * S)
        indices_prev = indices + (HS * S * 4)
        pole_indices = torch.asarray([icosphere.vertex_south_index, icosphere.vertex_north_index], device=device)

        self.inner = _create_weights(v_cart, base_dirs, 
                                    ((0,), (1,), (2, 3), (4, 5)),
                                    indices[1:-1, 1:-1],
                                    [
                                        indices[:-2, :-2],
                                        indices[2:, 2:],
                                        indices[:-2, 1:-1],
                                        indices[1:-1, 2:],
                                        indices[1:-1, :-2],
                                        indices[2:, 1:-1]
                                    ])
        
        self.east = _create_weights(v_cart, base_dirs,
                                   ((0,), (1,), (2, 3), (4, 5)),
                                   indices[0, 1:HS],
                                   [
                                       indices_next[HS-1, slice(HS, S-1)],  # weight 0 (wrap_prev)
                                       indices[1, 2:HS+1],                   # weight 1
                                       indices_next[HS-1, slice(HS+1, S)],   # weight 2 (wrap_prev)
                                       indices[0, 2:HS+1],                   # weight 3
                                       indices[0, 0:HS-1],                   # weight 4
                                       indices[1, 1:HS]                      # weight 5
                                   ])

        self.cn = _create_weights(v_cart, base_dirs,
                                ((0,), (1,), (2,), (3, 4)),
                                indices[0, HS],
                                [
                                    indices_next[HS-1, S-1],  # weight 0 (wrap_prev)
                                    indices[1, HS+1],         # weight 1
                                    indices[0, HS+1],         # weight 2
                                    indices[0, HS-1],         # weight 3
                                    indices[1, HS]            # weight 4
                                ])

        self.ne = _create_weights(v_cart, base_dirs,
                                ((0, 1), (2, 3), (4,), (5,)),
                                indices[0, HS+1:S-1],
                                [
                                    indices_prev[slice(HS-2, 0, -1), S-1],  # weight 0 (wrap_prev)
                                    indices_prev[slice(HS-1, 1, -1), S-1],  # weight 1 (wrap_prev_acc)
                                    indices[1, HS+2:S],                     # weight 2
                                    indices[1, HS+1:S-1],                   # weight 3
                                    indices[0, HS+2:S],                     # weight 4
                                    indices[0, HS:S-2]                      # weight 5
                                ])

        self.nne = _create_weights(v_cart, base_dirs,
                                ((0, 1), (2, 3), (4,), (5,)),
                                indices[0, S-1],
                                [
                                    indices_prev[0, S-1],       # weight 0 (wrap_prev)
                                    indices_prev[1, S-1],       # weight 1 (wrap_prev_acc)
                                    indices_next[0, S-1],       # weight 2 (wrap_next)
                                    indices[1, S-1],            # weight 3
                                    pole_indices[1],           # weight 4 (north pole)
                                    indices[0, S-2]             # weight 5
                                ])

        self.nw = _create_weights(v_cart, base_dirs,
                                ((0,), (1,), (2, 3), (4, 5)),
                                indices[1:HS-1, S-1],
                                [
                                    indices[0:HS-2, S-2],                    # weight 0
                                    indices_next[0, slice(S-2, HS, -1)],     # weight 1 (wrap_next)
                                    indices[0:HS-2, S-1],                    # weight 2
                                    indices_next[0, slice(S-1, HS+1, -1)],   # weight 3 (wrap_next_acc)
                                    indices[1:HS-1, S-2],                    # weight 4
                                    indices[2:HS, S-1]                       # weight 5
                                ])

        self.nww = _create_weights(v_cart, base_dirs,
                                ((0,), (1,), (2, 3), (4, 5)),
                                indices[HS-1, S-1],
                                [
                                    indices[HS-2, S-2],        # weight 0
                                    indices_next[0, HS],       # weight 1 (wrap_next)
                                    indices[HS-2, S-1],        # weight 2
                                    indices_next[0, HS+1],     # weight 3 (wrap_next_acc)
                                    indices[HS-1, S-2],        # weight 4
                                    indices_next[0, HS-1]      # weight 5 (wrap_next_acc)
                                ])

        self.west = _create_weights(v_cart, base_dirs,
                                ((0,), (1,), (2, 3), (4, 5)),
                                indices[HS-1, HS:S-1],
                                [
                                    indices[HS-2, HS-1:S-2],              # weight 0
                                    indices_next[0, slice(1, HS)],        # weight 1 (wrap_next)
                                    indices[HS-2, HS:S-1],                # weight 2
                                    indices[HS-1, HS+1:S],                # weight 3
                                    indices[HS-1, HS-1:S-2],              # weight 4
                                    indices_next[0, slice(0, HS-1)]       # weight 5 (wrap_next_acc)
                                ])

        self.sw = _create_weights(v_cart, base_dirs,
                                ((0,), (1,), (2, 3), (4, 5)),
                                indices[HS-1, 1:HS],
                                [
                                    indices[HS-2, 0:HS-1],                    # weight 0
                                    indices_next[slice(HS-2, -1, -1), 0],     # weight 1 (wrap_next)
                                    indices[HS-2, 1:HS],                      # weight 2
                                    indices[HS-1, 2:HS+1],                    # weight 3
                                    indices[HS-1, 0:HS-1],                    # weight 4
                                    indices_next[slice(HS-1, 0, -1), 0]       # weight 5 (wrap_next_acc)
                                ])

        self.sse = _create_weights(v_cart, base_dirs,
                                ((0,), (1,), (2, 3), (4,)),
                                indices[HS-1, 0],
                                [
                                    indices_prev[HS-1, 0],   # weight 0 (wrap_prev)
                                    indices_next[HS-1, 0],   # weight 1 (wrap_next)
                                    indices[HS-2, 0],        # weight 2
                                    indices[HS-1, 1],        # weight 3
                                    pole_indices[0]         # weight 4 (south pole)
                                ])

        self.se = _create_weights(v_cart, base_dirs,
                                ((0, 1), (2, 3), (4,), (5,)),
                                indices[1:HS-1, 0],
                                [
                                    indices_prev[HS-1, slice(HS-1, 1, -1)],   # weight 0 (wrap_prev)
                                    indices_prev[HS-1, slice(HS-2, 0, -1)],   # weight 1 (wrap_prev_acc)
                                    indices[1:HS-1, 1],                       # weight 2
                                    indices[2:HS, 1],                         # weight 3
                                    indices[0:HS-2, 0],                       # weight 4
                                    indices[2:HS, 0]                          # weight 5
                                ])

        self.cs = _create_weights(v_cart, base_dirs,
                                ((0,), (1,), (2, 3), (4,)),
                                indices[0, 0],
                                [
                                    indices_prev[HS-1, HS-1],   # weight 0 (wrap_prev)
                                    indices[1, 1],              # weight 1
                                    indices_prev[HS-1, HS],     # weight 2 (wrap_prev)
                                    indices[0, 1],              # weight 3
                                    indices[1, 0]               # weight 4
                                ])


class NeighbourWeightTableCache:
    def __init__(self):
        self.cache = defaultdict(dict)
    
    def get_table(self, level, device='cpu'):
        dev_idx = torch.device(device).index

        if dev_idx in self.cache[level]:
            return self.cache[level][dev_idx]

        table = NeighbourWeightTable(level, device)

        self.cache[level][dev_idx] = table

        return table

_GLOBAL_WEIGHT_TABLES = NeighbourWeightTableCache()

#@torch.jit.script
def _wrap_prev_diff(output, input_full, input, j, i, weights):
    output[:, 1:, ...] = (input_full[:, :4, j, i] - input[:, 1:, ...]) * weights
    output[:, 0, ...] = (input_full[:, 4, j, i] - input[:, 0, ...]) * weights

#@torch.jit.script
def _wrap_prev_diff_acc(output, input_full, input, j, i, weights):
    output[:, 1:, ...] += (input_full[:, :4, j, i] - input[:, 1:, ...]) * weights
    output[:, 0, ...] += (input_full[:, 4, j, i] - input[:, 0, ...]) * weights

#@torch.jit.script
def _wrap_next_diff(output, input_full, input, j, i, weights):
    output[:, :4, ...] = (input_full[:, 1:, j, i] - input[:, :4, ...]) * weights
    output[:, 5, ...] = (input_full[:, 0, j, i] - input[:, 4, ...]) * weights

#@torch.jit.script
def _wrap_next_diff_acc(output, input_full, input, j, i, weights):
    output[:, :4, ...] += (input_full[:, 1:, j, i] - input[:, :4, ...]) * weights
    output[:, 5, ...] += (input_full[:, 0, j, i] - input[:, 4, ...]) * weights

#@torch.jit.script
def _get_gradient_impl(L, input, poles, weight_table, output):
    HS, S = utils.get_face_side_dims(L)
    # input [C, face_side, width, height]
    # output [C, dir, face_side, width, height]
    # poles [C, 2]

    # Inner
    inner_in = input[..., 1:HS-1, 1:S-1]
    inner_out = output[..., 1:HS-1, 1:S-1]
    inner_out[:, 0, ...] = (input[:, :, :HS-2, :S-2] - inner_in) * weight_table.inner[None, None, 0, ...] #east
    inner_out[:, 1, ...] = (input[:, :, 2:HS, 2:S] - inner_in) * weight_table.inner[None, None, 1, ...] # west
    inner_out[:, 2, ...] = (input[:, :, :HS-2, 1:S-1] - inner_in) * weight_table.inner[None, None, 2, ...]   # north
    inner_out[:, 2, ...] += (input[:, :, 1:HS-1, 2:S] - inner_in) * weight_table.inner[None, None, 3, ...]
    inner_out[:, 3, ...] = (input[:, :, 1:HS-1, :S-2] - inner_in) * weight_table.inner[None, None, 4, ...]  # east
    inner_out[:, 3, ...] += (input[:, :, 2:HS, 1:S-1] - inner_in) * weight_table.inner[None, None, 5, ...]

    # east
    east_in = input[..., 0, 1:HS]
    east_out = output[..., 0, 1:HS]
    _wrap_prev_diff(east_out[:, 1, ...], input, east_in, HS-1, slice(HS,S-1), weight_table.east[None, None, 0, ...])
    east_out[:, 1, ...] = (input[:, :, 1, 2:HS+1] - east_in) * weight_table.east[None, None, 1, ...]
    _wrap_prev_diff(east_out[:, 2, ...], input, east_in, HS-1, slice(HS+1, S), weight_table.east[None, None, 2, ...])
    east_out[:, 2, ...] += (input[:, :, 0, 2:HS+1] - east_in) * weight_table.east[None, None, 3, ...]
    east_out[:, 3, ...] = (input[:, :, 0, 0:HS-1] - east_in) * weight_table.east[None, None, 4, ...]
    east_out[:, 3, ...] += (input[:, :, 1, 1:HS] - east_in) * weight_table.east[None, None, 5, ...]

    # corner north
    cn_in = input[..., 0, HS]
    cn_out = output[..., 0, HS]
    _wrap_prev_diff(cn_out[:, 0, :], input, cn_in, HS-1, S-1, weight_table.cn[None, None, 0])
    cn_out[:, 1, :] = (input[:, :, 1, HS+1] - cn_in) * weight_table.cn[None, None, 1]
    cn_out[:, 2, :] = (input[:, :, 0, HS+1] - cn_in) * weight_table.cn[None, None, 2]
    cn_out[:, 3, :] = (input[:, :, 0, HS-1] - cn_in) * weight_table.cn[None, None, 3]
    cn_out[:, 3, :] += (input[:, :, 1, HS] - cn_in) * weight_table.cn[None, None, 4]

    # ne
    ne_in = input[..., 0, HS+1:S-1]
    ne_out = output[..., 0, HS+1:S-1]
    _wrap_prev_diff(ne_out[:, 0, ...], input, ne_in, slice(HS-2, 0, -1), S-1, weight_table.ne[None, None, 0, :])
    _wrap_prev_diff_acc(ne_out[:, 0, ...], input, ne_in, slice(HS-1, 1, -1), S-1, weight_table.ne[None, None, 1, :])
    ne_out[:, 1, ...] = (input[:, :, 1, HS+2:S] - ne_in) * weight_table.ne[None, None, 2, :]
    ne_out[:, 1, ...] += (input[:, :, 1, HS+1:S-1] - ne_in) * weight_table.ne[None, None, 3, :]
    ne_out[:, 2, ...] = (input[:, :, 0, HS+2:S] - ne_in) * weight_table.ne[None, None, 4, :]
    ne_out[:, 3, ...] = (input[:, :, 0, HS:S-2] - ne_in) * weight_table.ne[None, None, 5, :]

    # nne
    nne_in = input[..., 0, S-1]
    nne_out = output[..., 0, S-1]
    _wrap_prev_diff(nne_out[:, 0, :], input, nne_in, 0, S-1, weight_table.nne[None, None, 0])
    _wrap_prev_diff_acc(nne_out[:, 0, :], input, nne_in, 1, S-1, weight_table.nne[None, None, 1])
    _wrap_next_diff(nne_out[:, 1, :], input, nne_in, 0, S-1, weight_table.nne[None, None, 2])
    nne_out[:, 1, :] = (input[:, :, 1, S-1] - nne_in) * weight_table.nne[None, None, 3]
    nne_out[:, 2, :] = (poles[:, 1] - nne_in) * weight_table.nne[None, None, 4]
    nne_out[:, 3, :] = (input[:, :, 0, S-2] - nne_in) * weight_table.nne[None, None, 5]

    # nw
    nw_in = input[..., 1:HS-1, S-1]
    nw_out = output[..., 1:HS-1, S-1]
    nw_out[:, 0, ...] = (input[:, :, 0:HS-2, S-2] - nw_in) * weight_table.nw[None, None, 0, :]
    _wrap_next_diff(nw_out[:, 1, ...], input, nw_in, 0, slice(S-2,HS,-1), weight_table.nw[None, None, 1, :])
    nw_out[:, 2, ...] = (input[:, :, 0:HS-2, S-1] - nw_in) * weight_table.nw[None, None, 2, :]
    _wrap_next_diff_acc(nw_out[:, 2, ...], input, nw_in, 0, slice(S-1,HS+1,-1), weight_table.nw[None, None, 3, :])
    nw_out[:, 3, ...] = (input[:, :, 1:HS-1, S-2] - nw_in) * weight_table.nw[None, None, 4, :]
    nw_out[:, 3, ...] += (input[:, :, 2:HS, S-1] - nw_in) * weight_table.nw[None, None, 5, :]

    # nww
    nww_in = input[..., HS-1, S-1]
    nww_out = output[..., HS-1, S-1]
    nww_out[:, 0, :] = (input[:, :, HS-2, S-2] - nww_in) * weight_table.nww[None, None, 0]
    _wrap_next_diff(nww_out[:, 1, :], input, nww_in, 0, HS, weight_table.nww[None, None, 1])
    nww_out[:, 2, :] = (input[:, :, HS-2, S-1] - nww_in) * weight_table.nww[None, None, 2]
    _wrap_next_diff_acc(nww_out[:, 2, :], input, nww_in, 0, HS+1, weight_table.nww[None, None, 3])
    nww_out[:, 3, :] = (input[:, :, HS-1, S-2] - nww_in) * weight_table.nww[None, None, 4]
    _wrap_next_diff_acc(nww_out[:, 3, :], input, nww_in, 0, HS-1, weight_table.nww[None, None, 5])

    #west
    west_in = input[..., HS-1, HS:S-1]
    west_out = output[..., HS-1, HS:S-1]
    west_out[:, 0, ...] = (input[:, :, HS-2, HS-1:S-2] - west_in) * weight_table.west[None, None, 0, :]
    _wrap_next_diff(west_out[:, 1, ...], input, west_in, 0, slice(1, HS), weight_table.west[None, None, 1, :])
    west_out[:, 2, ...] = (input[:, :, HS-2, HS:S-1] - west_in) * weight_table.west[None, None, 2, :]
    west_out[:, 2, ...] += (input[:, :, HS-1, HS+1:S] - west_in) * weight_table.west[None, None, 3, :]
    west_out[:, 3, ...] = (input[:, :, HS-1, HS-1:S-2] - west_in) * weight_table.west[None, None, 4, :]
    _wrap_next_diff_acc(west_out[:, 3, ...], input, west_in, 0, slice(0, HS-1), weight_table.west[None, None, 5, :])

    # SW
    sw_in = input[..., HS-1, 1:HS]
    sw_out = output[..., HS-1, 1:HS]
    sw_out[:, 0, ...] = (input[:, :, HS-2, 0:HS-1] - sw_in) * weight_table.sw[None, None, 0, :]
    _wrap_next_diff(sw_out[:, 1, ...], input, sw_in, slice(HS-2, -1, -1), 0, weight_table.sw[None, None, 1, :])
    sw_out[:, 2, ...] = (input[:, :, HS-2, 1:HS] - sw_in) * weight_table.sw[None, None, 2, :]
    sw_out[:, 2, ...] += (input[:, :, HS-1, 2:HS+1] - sw_in) * weight_table.sw[None, None, 3, :]
    sw_out[:, 3, ...] = (input[:, :, HS-1, 0:HS-1] - sw_in) * weight_table.sw[None, None, 4, :]
    _wrap_next_diff_acc(sw_out[:, 3, ...], input, sw_in, slice(HS-1, 0, -1), 0, weight_table.sw[None, None, 5, :])
    
    # SSE
    sse_in = input[..., HS-1, 0]
    sse_out = output[..., HS-1, 0]
    _wrap_prev_diff(sse_out[:, 0, :], input, sse_in, HS-1, 0, weight_table.sse[None, None, 0])
    _wrap_next_diff(sse_out[:, 1, :], input, sse_in, HS-1, 0, weight_table.sse[None, None, 1])
    sse_out[:, 2, :] = (input[:, :, HS-2, 0] - sse_in) * weight_table.sse[None, None, 2]
    sse_out[:, 2, :] += (input[:, :, HS-1, 1] - sse_in) * weight_table.sse[None, None, 3]
    sse_out[:, 3, :] = (poles[:, 0] - sse_in) * weight_table.sse[None, None, 4]

    # SE
    se_in = input[..., 1:HS-1, 0]
    se_out = output[..., 1:HS-1, 0]
    _wrap_prev_diff(se_out[:, 0, ...], input, se_in, HS-1, slice(HS-1,1,-1), weight_table.se[None, None, 0, :])
    _wrap_prev_diff_acc(se_out[:, 0, ...], input, se_in, HS-1, slice(HS-2,0,-1), weight_table.se[None, None, 1, :])
    se_out[:, 1, ...] = (input[:, :, 1:HS-1, 1] - se_in) * weight_table.se[None, None, 2, :]
    se_out[:, 1, ...] += (input[:, :, 2:HS, 1] - se_in) * weight_table.se[None, None, 3, :]
    se_out[:, 2, ...] = (input[:, :, 0:HS-2, 0] - se_in) * weight_table.se[None, None, 4, :]
    se_out[:, 3, ...] = (input[:, :, 2:HS, 0] - se_in) * weight_table.se[None, None, 5, :]

    # CS
    cs_in = input[..., 0, 0]
    cs_out = output[..., 0, 0]
    _wrap_prev_diff(cs_out[:, 0, ...], input, cs_in, HS-1, HS-1, weight_table.cs[None, None, 0, :])
    cs_out[:, 1, :] = (input[:, :, 1, 1] - cs_in) * weight_table.cs[None, None, 1, :]
    _wrap_prev_diff(cs_out[:, 2, ...], input, cs_in, HS-1, HS, weight_table.cs[None, None, 2, :])
    cs_out[:, 2, :] += (input[:, :, 0, 1] - cs_in) * weight_table.cs[None, None, 3, :]
    cs_out[:, 3, :] = (input[:, :, 1, 0] - cs_in) * weight_table.cs[None, None, 4, :]


class IcosphereGradient(nn.Module):
    def __init__(self, keep_original_value = False):
        super().__init__()

        self.keep_original_value = keep_original_value

    def forward(self, x):
        assert(len(x.shape) >= 1 and len(x.shape) <= 3)
        input_dim = len(x.shape)
        
        for _ in range(3 - input_dim):
            x = x.unsqueeze(0)  # unsqueeze N and C dims

        N, C, V = x.shape
        L = utils.get_icosphere_level(V)
        w, h = utils.get_face_side_dims(L)

        print(f'{L} {w} {h}')

        non_polar = x[..., :-2].view(N*C, 5, w, h)
        poles = x[..., -2:].view(N*C, 2)

        weight_table = _GLOBAL_WEIGHT_TABLES.get_table(L, x.device)

        if self.keep_original_value:
            output = torch.empty((N, C, 5, V), device=x.device, dtype=x.dtype)
            output[:, :, :, 4] = x
            out_view = output.view(N*C, 5, 5, w, h)
        else:
            output = torch.empty((N, C, 4, V), device=x.device, dtype=x.dtype)
            out_view = output.view(N*C, 4, 5, w, h)

        _get_gradient_impl(L, non_polar, poles, weight_table, out_view)

        for _ in range(3 - input_dim):
            output = output.squeeze(0)

        return output