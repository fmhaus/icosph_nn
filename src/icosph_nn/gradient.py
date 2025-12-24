import torch.nn as nn
import torch
from icosph_nn import utils
from icosph_nn.icosphere import Icosphere, cart2sph, sph2cart
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
            to_n = vertices[neighbour_indices[dir_group[0]]] - vertices[indices]
            dot = torch.sum(to_n * base_vec[indices], dim=-1)
            results.append(1 / dot)
        else:
            assert(False)
    
    return torch.stack(results, dim=0)

class NeighbourWeightTable:
    def __init__(self, L, device):
        HS, S = utils.get_face_side_dims(L)

        icosphere = Icosphere(L, device)
        v_cart = icosphere.generate_vertices()
        V, _ = v_cart.shape
        
        N = (V-2) // 5
        v_sph = cart2sph(v_cart[:N])    
    

        north = Icosphere.get_north_directions(v_sph)
        east = Icosphere.get_east_directions(v_sph)
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
                                    indices_prev[HS-1, HS:S-1],  # weight 0 (wrap_prev)
                                    indices[1, 2:HS+1],          # weight 1
                                    indices_prev[HS-1, HS+1:S],  # weight 2 (wrap_prev)
                                    indices[0, 2:HS+1],          # weight 3
                                    indices[0, 0:HS-1],          # weight 4
                                    indices[1, 1:HS]             # weight 5
                                ])

        self.cn = _create_weights(v_cart, base_dirs,
                                ((0,), (1,), (2,), (3, 4)),
                                indices[0, HS],
                                [
                                    indices_prev[HS-1, S-1],  # weight 0 (wrap_prev)
                                    indices[1, HS+1],         # weight 1
                                    indices[0, HS+1],         # weight 2
                                    indices[0, HS-1],         # weight 3
                                    indices[1, HS]            # weight 4
                                ])

        self.ne = _create_weights(v_cart, base_dirs,
                                ((0, 1), (2, 3), (4,), (5,)),
                                indices[0, HS+1:S-1],
                                [
                                    indices_prev[1:HS-1, S-1].flip(0),  # rows flipped (1D)
                                    indices_prev[2:HS, S-1].flip(0),    # rows flipped (1D)
                                    indices[1, HS+2:S],                 # weight 2
                                    indices[1, HS+1:S-1],               # weight 3
                                    indices[0, HS+2:S],                 # weight 4
                                    indices[0, HS:S-2]                  # weight 5
                                ])

        self.nne = _create_weights(v_cart, base_dirs,
                                ((0, 1), (2, 3), (4,), (5,)),
                                indices[0, S-1],
                                [
                                    indices_prev[0, S-1],       # weight 0 (wrap_prev)
                                    indices_prev[1, S-1],       # weight 1 (wrap_prev_acc)
                                    indices_next[0, S-1],       # weight 2 (wrap_next)
                                    indices[1, S-1],            # weight 3
                                    pole_indices[1],            # weight 4 (north pole)
                                    indices[0, S-2]             # weight 5
                                ])

        self.nw = _create_weights(v_cart, base_dirs,
                                ((0,), (1,), (2, 3), (4, 5)),
                                indices[1:HS-1, S-1],
                                [
                                    indices[0:HS-2, S-2],                    # weight 0
                                    indices_next[0, HS+1:S-1].flip(-1),      # columns flipped (1D)
                                    indices[0:HS-2, S-1],                    # weight 2
                                    indices_next[0, HS+2:S].flip(-1),        # columns flipped (1D)
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
                                    indices[HS-2, HS-1:S-2],  # weight 0
                                    indices_next[0, 1:HS],    # weight 1 (wrap_next)
                                    indices[HS-2, HS:S-1],    # weight 2
                                    indices[HS-1, HS+1:S],    # weight 3
                                    indices[HS-1, HS-1:S-2],  # weight 4
                                    indices_next[0, 0:HS-1]   # weight 5 (wrap_next_acc)
                                ])

        self.sw = _create_weights(v_cart, base_dirs,
                                ((0,), (1,), (2, 3), (4, 5)),
                                indices[HS-1, 1:HS],
                                [
                                    indices[HS-2, 0:HS-1],          # weight 0
                                    indices_next[0:HS-1, 0].flip(0), # rows flipped (1D)
                                    indices[HS-2, 1:HS],            # weight 2
                                    indices[HS-1, 2:HS+1],          # weight 3
                                    indices[HS-1, 0:HS-1],          # weight 4
                                    indices_next[1:HS, 0].flip(0)   # rows flipped (1D)
                                ])

        self.sse = _create_weights(v_cart, base_dirs,
                                ((0,), (1,), (2, 3), (4,)),
                                indices[HS-1, 0],
                                [
                                    indices_prev[HS-1, 0],   # weight 0 (wrap_prev)
                                    indices_next[HS-1, 0],   # weight 1 (wrap_next)
                                    indices[HS-2, 0],        # weight 2
                                    indices[HS-1, 1],        # weight 3
                                    pole_indices[0]          # weight 4 (south pole)
                                ])

        self.se = _create_weights(v_cart, base_dirs,
                                ((0, 1), (2, 3), (4,), (5,)),
                                indices[1:HS-1, 0],
                                [
                                    indices_prev[HS-1, 2:HS].flip(-1),    # columns flipped (1D)
                                    indices_prev[HS-1, 1:HS-1].flip(-1),  # columns flipped (1D)
                                    indices[1:HS-1, 1],                  # weight 2
                                    indices[2:HS, 1],                    # weight 3
                                    indices[0:HS-2, 0],                  # weight 4
                                    indices[2:HS, 0]                     # weight 5
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

def _normalize_angle_diff(angles):
    return (angles + torch.pi) % (2 * torch.pi) - torch.pi

def _create_weights_sph(v_sph, base_dirs, weight_grouping, indices, neighbour_indices):
    results = []
    for dir_group, base_vec in zip(weight_grouping, base_dirs):
        if len(dir_group) == 2:
            to_n1 = v_sph[neighbour_indices[dir_group[0]]] - v_sph[indices]
            to_n2 = v_sph[neighbour_indices[dir_group[1]]] - v_sph[indices]
            
            to_n1[..., 0] = _normalize_angle_diff(to_n1[..., 0])
            to_n2[..., 0] = _normalize_angle_diff(to_n2[..., 0])
            
            p1 = torch.sum(to_n1 * base_vec, dim=-1)
            p2 = torch.sum(to_n2 * base_vec, dim=-1)
            q1 = torch.sum(to_n1 * base_vec.flip(0), dim=-1)
            q2 = torch.sum(to_n2 * base_vec.flip(0), dim=-1)

            """
            k = p1 * q2 - p2 * q1
            results.append(q2 / k)
            results.append(-q1 / k)
            """
            k = p1**2 + p2**2
            results.append(p1 / k)
            results.append(p2 / k)

        elif len(dir_group) == 1:
            to_n = v_sph[neighbour_indices[dir_group[0]]] - v_sph[indices]
            to_n[..., 0] = _normalize_angle_diff(to_n[..., 0])
            p = torch.sum(to_n * base_vec, dim=-1)
            results.append(1 / p)
        else:
            assert(False)
    
    return torch.stack(results, dim=0)

class NeighbourWeightTableSph:
    def __init__(self, L, device):
        HS, S = utils.get_face_side_dims(L)

        icosphere = Icosphere(L, device)
        v_sph = icosphere.generate_vertices(cartesian=False)
        V, _ = v_sph.shape
        
        N = (V-2) // 5

        indices = torch.arange(N, device=device).view(HS, S)
        indices_next = indices + (HS * S)
        indices_prev = indices + (HS * S * 4)

        pole_indices = torch.asarray([icosphere.vertex_south_index, icosphere.vertex_north_index], device=device)

        base_dirs = [
            torch.asarray((1, 0), device=device),
            torch.asarray((-1, 0), device=device),
            torch.asarray((0, 1), device=device),
            torch.asarray((0, -1), device=device),
        ]

        self.inner = _create_weights_sph(v_sph, base_dirs, 
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

        self.east = _create_weights_sph(v_sph, base_dirs,
                                ((0,), (1,), (2, 3), (4, 5)),
                                indices[0, 1:HS],
                                [
                                    indices_prev[HS-1, HS:S-1],  # weight 0 (wrap_prev)
                                    indices[1, 2:HS+1],          # weight 1
                                    indices_prev[HS-1, HS+1:S],  # weight 2 (wrap_prev)
                                    indices[0, 2:HS+1],          # weight 3
                                    indices[0, 0:HS-1],          # weight 4
                                    indices[1, 1:HS]             # weight 5
                                ])

        self.cn = _create_weights_sph(v_sph, base_dirs,
                                ((0,), (1,), (2,), (3, 4)),
                                indices[0, HS],
                                [
                                    indices_prev[HS-1, S-1],  # weight 0 (wrap_prev)
                                    indices[1, HS+1],         # weight 1
                                    indices[0, HS+1],         # weight 2
                                    indices[0, HS-1],         # weight 3
                                    indices[1, HS]            # weight 4
                                ])

        self.ne = _create_weights_sph(v_sph, base_dirs,
                                ((0, 1), (2, 3), (4,), (5,)),
                                indices[0, HS+1:S-1],
                                [
                                    indices_prev[1:HS-1, S-1].flip(0),  # rows flipped (1D)
                                    indices_prev[2:HS, S-1].flip(0),    # rows flipped (1D)
                                    indices[1, HS+2:S],                 # weight 2
                                    indices[1, HS+1:S-1],               # weight 3
                                    indices[0, HS+2:S],                 # weight 4
                                    indices[0, HS:S-2]                  # weight 5
                                ])

        self.nne = _create_weights_sph(v_sph, base_dirs,
                                ((0, 1), (2, 3), (4,), (5,)),
                                indices[0, S-1],
                                [
                                    indices_prev[0, S-1],       # weight 0 (wrap_prev)
                                    indices_prev[1, S-1],       # weight 1 (wrap_prev_acc)
                                    indices_next[0, S-1],       # weight 2 (wrap_next)
                                    indices[1, S-1],            # weight 3
                                    pole_indices[1],            # weight 4 (north pole)
                                    indices[0, S-2]             # weight 5
                                ])

        self.nw = _create_weights_sph(v_sph, base_dirs,
                                ((0,), (1,), (2, 3), (4, 5)),
                                indices[1:HS-1, S-1],
                                [
                                    indices[0:HS-2, S-2],                    # weight 0
                                    indices_next[0, HS+1:S-1].flip(-1),      # columns flipped (1D)
                                    indices[0:HS-2, S-1],                    # weight 2
                                    indices_next[0, HS+2:S].flip(-1),        # columns flipped (1D)
                                    indices[1:HS-1, S-2],                    # weight 4
                                    indices[2:HS, S-1]                       # weight 5
                                ])

        self.nww = _create_weights_sph(v_sph, base_dirs,
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

        self.west = _create_weights_sph(v_sph, base_dirs,
                                ((0,), (1,), (2, 3), (4, 5)),
                                indices[HS-1, HS:S-1],
                                [
                                    indices[HS-2, HS-1:S-2],  # weight 0
                                    indices_next[0, 1:HS],    # weight 1 (wrap_next)
                                    indices[HS-2, HS:S-1],    # weight 2
                                    indices[HS-1, HS+1:S],    # weight 3
                                    indices[HS-1, HS-1:S-2],  # weight 4
                                    indices_next[0, 0:HS-1]   # weight 5 (wrap_next_acc)
                                ])

        self.sw = _create_weights_sph(v_sph, base_dirs,
                                ((0,), (1,), (2, 3), (4, 5)),
                                indices[HS-1, 1:HS],
                                [
                                    indices[HS-2, 0:HS-1],          # weight 0
                                    indices_next[0:HS-1, 0].flip(0), # rows flipped (1D)
                                    indices[HS-2, 1:HS],            # weight 2
                                    indices[HS-1, 2:HS+1],          # weight 3
                                    indices[HS-1, 0:HS-1],          # weight 4
                                    indices_next[1:HS, 0].flip(0)   # rows flipped (1D)
                                ])

        self.sse = _create_weights_sph(v_sph, base_dirs,
                                ((0,), (1,), (2, 3), (4,)),
                                indices[HS-1, 0],
                                [
                                    indices_prev[HS-1, 0],   # weight 0 (wrap_prev)
                                    indices_next[HS-1, 0],   # weight 1 (wrap_next)
                                    indices[HS-2, 0],        # weight 2
                                    indices[HS-1, 1],        # weight 3
                                    pole_indices[0]          # weight 4 (south pole)
                                ])

        self.se = _create_weights_sph(v_sph, base_dirs,
                                ((0, 1), (2, 3), (4,), (5,)),
                                indices[1:HS-1, 0],
                                [
                                    indices_prev[HS-1, 2:HS].flip(-1),    # columns flipped (1D)
                                    indices_prev[HS-1, 1:HS-1].flip(-1),  # columns flipped (1D)
                                    indices[1:HS-1, 1],                  # weight 2
                                    indices[2:HS, 1],                    # weight 3
                                    indices[0:HS-2, 0],                  # weight 4
                                    indices[2:HS, 0]                     # weight 5
                                ])

        self.cs = _create_weights_sph(v_sph, base_dirs,
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

        table = NeighbourWeightTableSph(level, device)

        self.cache[level][dev_idx] = table

        return table

_GLOBAL_WEIGHT_TABLES = NeighbourWeightTableCache()

def _wrap_prev_diff(input, input_full, j, i, weights, rev=False):
    values_main = input_full[:, :4, j, i]
    values_wrap = input_full[:, 4, j, i]
    if rev:
        values_main = values_main.flip(-1)
        values_wrap = values_wrap.flip(-1)

    return torch.cat((
        (values_wrap - input[:, 0, ...]).unsqueeze(1) * weights,
        (values_main - input[:, 1:, ...]) * weights
        ), dim=1)

def _wrap_next_diff(input, input_full, j, i, weights, rev=False):
    values_main = input_full[:, 1:, j, i]
    values_wrap = input_full[:, 0, j, i]
    if rev:
        values_main = values_main.flip(-1)
        values_wrap = values_wrap.flip(-1)
    
    return torch.cat((
        (values_main - input[:, :4, ...]) * weights,
        (values_wrap - input[:, 4, ...]).unsqueeze(1) * weights
        ), dim=1)

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
    inner_out[:, 2, ...] = (input[:, :, :HS-2, 1:S-1] - inner_in) * weight_table.inner[None, None, 2, ...] +\
                           (input[:, :, 1:HS-1, 2:S] - inner_in) * weight_table.inner[None, None, 3, ...]  # north
    inner_out[:, 3, ...] = (input[:, :, 1:HS-1, :S-2] - inner_in) * weight_table.inner[None, None, 4, ...] +\
                           (input[:, :, 2:HS, 1:S-1] - inner_in) * weight_table.inner[None, None, 5, ...]  # south

    # east
    east_in = input[..., 0, 1:HS]
    east_out = output[..., 0, 1:HS]
    east_out[:, 0, ...] = _wrap_prev_diff(east_in, input, HS-1, slice(HS,S-1), weight_table.east[None, None, 0, ...])
    east_out[:, 1, ...] = (input[:, :, 1, 2:HS+1] - east_in) * weight_table.east[None, None, 1, ...]
    east_out[:, 2, ...] = _wrap_prev_diff(east_in, input, HS-1, slice(HS+1, S), weight_table.east[None, None, 2, ...]) +\
                          (input[:, :, 0, 2:HS+1] - east_in) * weight_table.east[None, None, 3, ...]
    east_out[:, 3, ...] = (input[:, :, 0, 0:HS-1] - east_in) * weight_table.east[None, None, 4, ...] +\
                          (input[:, :, 1, 1:HS] - east_in) * weight_table.east[None, None, 5, ...]

    # corner north
    cn_in = input[..., 0, HS]
    cn_out = output[..., 0, HS]
    cn_out[:, 0, :] = _wrap_prev_diff(cn_in, input, HS-1, S-1, weight_table.cn[None, None, 0])
    cn_out[:, 1, :] = (input[:, :, 1, HS+1] - cn_in) * weight_table.cn[None, None, 1]
    cn_out[:, 2, :] = (input[:, :, 0, HS+1] - cn_in) * weight_table.cn[None, None, 2]
    cn_out[:, 3, :] = (input[:, :, 0, HS-1] - cn_in) * weight_table.cn[None, None, 3] +\
                      (input[:, :, 1, HS] - cn_in) * weight_table.cn[None, None, 4]

    # ne
    ne_in = input[..., 0, HS+1:S-1]
    ne_out = output[..., 0, HS+1:S-1]
    ne_out[:, 0, ...] = _wrap_prev_diff(ne_in, input, slice(1, HS-1), S-1, weight_table.ne[None, None, 0, :], rev=True) +\
                        _wrap_prev_diff(ne_in, input, slice(2, HS), S-1, weight_table.ne[None, None, 1, :], rev=True)
    ne_out[:, 1, ...] = (input[:, :, 1, HS+2:S] - ne_in) * weight_table.ne[None, None, 2, :] +\
                        (input[:, :, 1, HS+1:S-1] - ne_in) * weight_table.ne[None, None, 3, :]
    ne_out[:, 2, ...] = (input[:, :, 0, HS+2:S] - ne_in) * weight_table.ne[None, None, 4, :]
    ne_out[:, 3, ...] = (input[:, :, 0, HS:S-2] - ne_in) * weight_table.ne[None, None, 5, :]

    # nne
    nne_in = input[..., 0, S-1]
    nne_out = output[..., 0, S-1]
    nne_out[:, 0, :] = _wrap_prev_diff(nne_in, input, 0, S-1, weight_table.nne[None, None, 0]) +\
                       _wrap_prev_diff(nne_in, input, 1, S-1, weight_table.nne[None, None, 1])
    nne_out[:, 1, :] = _wrap_next_diff(nne_in, input, 0, S-1, weight_table.nne[None, None, 2]) +\
                        (input[:, :, 1, S-1] - nne_in) * weight_table.nne[None, None, 3]
    nne_out[:, 2, :] = (poles[:, 1] - nne_in) * weight_table.nne[None, None, 4]
    nne_out[:, 3, :] = (input[:, :, 0, S-2] - nne_in) * weight_table.nne[None, None, 5]

    # nw
    nw_in = input[..., 1:HS-1, S-1]
    nw_out = output[..., 1:HS-1, S-1]
    nw_out[:, 0, ...] = (input[:, :, 0:HS-2, S-2] - nw_in) * weight_table.nw[None, None, 0, :]
    nw_out[:, 1, ...] = _wrap_next_diff(nw_in, input, 0, slice(HS+1, S-1), weight_table.nw[None, None, 1, :], rev=True)
    nw_out[:, 2, ...] = (input[:, :, 0:HS-2, S-1] - nw_in) * weight_table.nw[None, None, 2, :] +\
                        _wrap_next_diff(nw_in, input, 0, slice(HS+2, S), weight_table.nw[None, None, 3, :], rev=True)
    nw_out[:, 3, ...] = (input[:, :, 1:HS-1, S-2] - nw_in) * weight_table.nw[None, None, 4, :] +\
                        (input[:, :, 2:HS, S-1] - nw_in) * weight_table.nw[None, None, 5, :]

    # nww
    nww_in = input[..., HS-1, S-1]
    nww_out = output[..., HS-1, S-1]    
    nww_out[:, 0, :] = (input[:, :, HS-2, S-2] - nww_in) * weight_table.nww[None, None, 0]
    nww_out[:, 1, :] = _wrap_next_diff(nww_in, input, 0, HS, weight_table.nww[None, None, 1])
    nww_out[:, 2, :] = (input[:, :, HS-2, S-1] - nww_in) * weight_table.nww[None, None, 2] +\
                       _wrap_next_diff(nww_in, input, 0, HS+1, weight_table.nww[None, None, 3])
    nww_out[:, 3, :] = (input[:, :, HS-1, S-2] - nww_in) * weight_table.nww[None, None, 4] +\
                       _wrap_next_diff(nww_in, input, 0, HS-1, weight_table.nww[None, None, 5])

    # west
    west_in = input[..., HS-1, HS:S-1]
    west_out = output[..., HS-1, HS:S-1]
    west_out[:, 0, ...] = (input[:, :, HS-2, HS-1:S-2] - west_in) * weight_table.west[None, None, 0, :]
    west_out[:, 1, ...] = _wrap_next_diff(west_in, input, 0, slice(1, HS), weight_table.west[None, None, 1, :])
    west_out[:, 2, ...] = (input[:, :, HS-2, HS:S-1] - west_in) * weight_table.west[None, None, 2, :] +\
                          (input[:, :, HS-1, HS+1:S] - west_in) * weight_table.west[None, None, 3, :]
    west_out[:, 3, ...] = (input[:, :, HS-1, HS-1:S-2] - west_in) * weight_table.west[None, None, 4, :] +\
                          _wrap_next_diff(west_in, input, 0, slice(0, HS-1), weight_table.west[None, None, 5, :])

    # SW
    sw_in = input[..., HS-1, 1:HS]
    sw_out = output[..., HS-1, 1:HS]
    sw_out[:, 0, ...] = (input[:, :, HS-2, 0:HS-1] - sw_in) * weight_table.sw[None, None, 0, :]
    sw_out[:, 1, ...] = _wrap_next_diff(sw_in, input, slice(0, HS-1), 0, weight_table.sw[None, None, 1, :], rev=True)
    sw_out[:, 2, ...] = (input[:, :, HS-2, 1:HS] - sw_in) * weight_table.sw[None, None, 2, :] +\
                        (input[:, :, HS-1, 2:HS+1] - sw_in) * weight_table.sw[None, None, 3, :]
    sw_out[:, 3, ...] = (input[:, :, HS-1, 0:HS-1] - sw_in) * weight_table.sw[None, None, 4, :] +\
                        _wrap_next_diff(sw_in, input, slice(1, HS), 0, weight_table.sw[None, None, 5, :], rev=True)

    # SSE
    sse_in = input[..., HS-1, 0]
    sse_out = output[..., HS-1, 0]
    sse_out[:, 0, :] = _wrap_prev_diff(sse_in, input, HS-1, 0, weight_table.sse[None, None, 0])
    sse_out[:, 1, :] = _wrap_next_diff(sse_in, input, HS-1, 0, weight_table.sse[None, None, 1])
    sse_out[:, 2, :] = (input[:, :, HS-2, 0] - sse_in) * weight_table.sse[None, None, 2] +\
                       (input[:, :, HS-1, 1] - sse_in) * weight_table.sse[None, None, 3]
    sse_out[:, 3, :] = (poles[:, 0] - sse_in) * weight_table.sse[None, None, 4]

    # SE
    se_in = input[..., 1:HS-1, 0]
    se_out = output[..., 1:HS-1, 0]
    se_out[:, 0, ...] = _wrap_prev_diff(se_in, input, HS-1, slice(2, HS), weight_table.se[None, None, 0, :], rev=True) +\
                        _wrap_prev_diff(se_in, input, HS-1, slice(1, HS-1), weight_table.se[None, None, 1, :], rev=True)
    se_out[:, 1, ...] = (input[:, :, 1:HS-1, 1] - se_in) * weight_table.se[None, None, 2, :] +\
                        (input[:, :, 2:HS, 1] - se_in) * weight_table.se[None, None, 3, :]
    se_out[:, 2, ...] = (input[:, :, 0:HS-2, 0] - se_in) * weight_table.se[None, None, 4, :]
    se_out[:, 3, ...] = (input[:, :, 2:HS, 0] - se_in) * weight_table.se[None, None, 5, :]

    # CS
    cs_in = input[..., 0, 0]
    cs_out = output[..., 0, 0]
    cs_out[:, 0, :] = _wrap_prev_diff(cs_in, input, HS-1, HS-1, weight_table.cs[None, None, 0])
    cs_out[:, 1, :] = (input[:, :, 1, 1] - cs_in) * weight_table.cs[None, None, 1]
    cs_out[:, 2, :] = _wrap_prev_diff(cs_in, input, HS-1, HS, weight_table.cs[None, None, 2]) +\
                      (input[:, :, 0, 1] - cs_in) * weight_table.cs[None, None, 3]
    cs_out[:, 3, :] = (input[:, :, 1, 0] - cs_in) * weight_table.cs[None, None, 4]

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

        non_polar = x[..., :-2].view(N*C, 5, w, h)
        poles = x[..., -2:].view(N*C, 2)

        weight_table = _GLOBAL_WEIGHT_TABLES.get_table(L, x.device)

        if self.keep_original_value:
            output = torch.empty((N, C, 5, V), device=x.device, dtype=x.dtype)
            output[:, :, :, 4] = x
            non_polar_out = output[..., :-2].view(N*C, 5, 5, w, h)
        else:
            output = torch.empty((N, C, 4, V), device=x.device, dtype=x.dtype)
            non_polar_out = output[..., :-2].view(N*C, 4, 5, w, h)

        _get_gradient_impl(L, non_polar, poles, weight_table, non_polar_out)

        for _ in range(3 - input_dim):
            output = output.squeeze(0)

        return output

def f():
    v = Icosphere(1)
    vs = v.generate_vertices()

    from icosph_nn.visuals import IcosphereVisualizer
    L = 5

    ico = Icosphere(L)

    colors = torch.empty((ico.get_vertex_count(), 3), dtype=torch.float32)
    for i, (lon,lat) in enumerate(ico.generate_vertices(cartesian=False, radians=False)):
        v = (lat + 90) / 180
        colors[i] = torch.asarray([v, v, v])

    vis = IcosphereVisualizer(L)
    vis.update_mesh(colors)

    grad_index = 0

    def on_key(key):
        nonlocal colors
        nonlocal vis
        nonlocal grad_index

        if key == b'g':
            grad = IcosphereGradient()

            print(f"Display grad: {grad_index}")
            g = grad(colors[:, 0])[grad_index, :-2]
            grad_index = (grad_index + 1) % 4

            scale = max(-torch.min(g), torch.max(g))
            print(f"Min: {torch.min(g).item()} Max: {torch.max(g).item()} Avg: {torch.mean(g).item()} Stdev: {torch.std(g).item()}")
            print(f"Arg: {torch.argmin(g).item()} {torch.argmax(g).item()}")
            
            #g = g / (scale + 0.0000001) / 2 + 0.5
            #print(scale)

            g = g / 2 + 0.5

            ncolors = torch.stack((g, g, g), dim=1)
            # ncolors[1, 0] = 1

            N = (Icosphere(L).get_vertex_count()-2) // 5
            HS, S = utils.get_face_side_dims(L)
            indices = torch.arange(N).view(HS, S)
            indices_next = indices + (HS * S)
            indices_prev = indices + (HS * S * 4)

            """
            se_in = input[..., 1:HS-1, 0]
            se_out = output[..., 1:HS-1, 0]
            _wrap_prev_diff(se_out[:, 0, ...], input, se_in, HS-1, slice(2, HS), weight_table.se[None, None, 0, :], rev=True)
            _wrap_prev_diff(se_out[:, 0, ...], input, se_in, HS-1, slice(1, HS-1), weight_table.se[None, None, 1, :], acc=True, rev=True)
            se_out[:, 1, ...] = (input[:, :, 1:HS-1, 1] - se_in) * weight_table.se[None, None, 2, :]
            se_out[:, 1, ...] += (input[:, :, 2:HS, 1] - se_in) * weight_table.se[None, None, 3, :]
            se_out[:, 2, ...] = (input[:, :, 0:HS-2, 0] - se_in) * weight_table.se[None, None, 4, :]
            se_out[:, 3, ...] = (input[:, :, 2:HS, 0] - se_in) * weight_table.se[None, None, 5, :]
            """

            ncolors = torch.cat((ncolors, torch.zeros(2, 3)), dim=0)

            vis.update_mesh(ncolors)
           

    vis.on_key_release = on_key
    vis.main_loop()

#f()