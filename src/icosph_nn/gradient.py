import torch.nn as nn
import torch
from icosph_nn import utils
from icosph_nn.icosphere import IcoSphere, cart2sph
from collections import defaultdict

class NeighbourWeightTable:
    def __init__(self, L, device):
        pass

class NeighbourWeightTableCache:
    def __init__(self):
        self.cache = defaultdict(dict)
    
    def get_table(self, level, device='cpu'):
        dev_idx = torch.device(device).index

        if dev_idx in self.cache[level]:
            return self.cache[level][dev_idx]

        table = self._create_table(level, device)

        self.cache[level][dev_idx] = table

        return table
    
    def _create_weights(self, vertices, indices, neighbour_indices, dir_indices, base_vectors):
        neighbour_vecs = vertices[neighbour_indices] - vertices[indices[:, None]]

        results = []
        for dir_index, base_vec in zip(dir_indices, base_vectors):
            dot = torch.sum(neighbour_vecs[:, dir_index, :] * base_vec[indices, None, :], dim=-1)
            if len(indices) == 1:
                dot_scaled = 1 / dot
            elif len(indices) == 2 and indices[0] == indices[1]:
                dot_scaled = 1 / dot[:, -1]
            else:
                dot_scaled = dot / (torch.sum(dot**2, dim=-1)).unsqueeze(-1)
            results.append(dot_scaled)
        
        return torch.cat(results, dim=1)
    
    def _basic_neighbours(self, V, indices, offsets):
        offsets = torch.asarray(offsets, device=indices.device)
        return (indices[:, None] + offsets[None, :]) % V

    def _strided_neighbours(self, indices, base, stride):
        base = torch.asarray(base, device=indices.device)[None, :]
        stride = torch.asarray(stride, device=indices.device)[None, :]
        return base + stride * torch.arange(len(indices), device=indices.device)[:, None]

    def _replace_pole_direction(self, indices, dir_index, pole_vertex):
        indices[-1, dir_index] = pole_vertex
        return indices

    def _create_table(self, level, device):
        # This needs to be benchmarked. 
        # Probably creating everything on CPU and then move result to GPU should be a lot faster because kernel launch overhead

        icosphere = IcoSphere(level, device)
        v_cart = icosphere.generate_vertices()
        V, _ = v_cart.shape
        
        N = (V-2) // 5
        v_sph = cart2sph(v_cart[:N])    

        north = IcoSphere.get_north_directions(v_sph)
        east = IcoSphere.get_east_directions(v_sph)

        n_rings_face = 2 ** level
        ring_size = 2 * n_rings_face
        S = ring_size

        indices = torch.arange(N, device=device).view(S//2, S)

        neighbours_base = self._basic_neighbours(V, indices[1:-1, 1:-1], (V-S-1, S+1, V-S, 1, V-1, S))
        neighbours_seam_east = self._basic_neighbours(V, indices[1:S//2], (V-S//2-1, S+1, V-S//2, 1, V-1, S))
        neighbours_seam_north_east = torch.concat((
                                        self._strided_neighbours(indices[S//2+1:S-1], (V-1, V-S-1), (-S,)),
                                        self._replace_pole_direction(
                                            self._basic_neighbours(V, indices[S//2+1:S-1], (S+1, S, 1, V-1)),
                                            2,
                                            icosphere.vertex_north_index)),
                                        dim=1)
        neighbours_seam_south_east = torch.concat((
                                        self._strided_neighbours(indices[S:S*S//2:S], (V-S//2-1, V-S//2-2), (-1,)),
                                        self._replace_pole_direction(
                                            self._basic_neighbours(V, indices[S:S*S//2:S], (1, S+1, V-S, S)), 
                                            3, 
                                            icosphere.vertex_south_index)), 
                                        dim=1)

        weights = self._create_weights(v_cart,
                                        indices, 
                                        neighbours_base,
                                        ((0,), (3,), (5, 4), (1, 2)), 
                                        (east, -east, north, -north))

        weights[1:S//2] = self._create_weights(v_cart,
                                        indices[1:S//2], 
                                        neighbours_seam_east,
                                        ((0,), (3,), (5, 4), (1, 2)), 
                                        (east, -east, north, -north))
        
        weights[S//2+1:S-1] = self._create_weights(v_cart,
                                        indices[S//2+1:S-1],
                                        neighbours_seam_north_east,
                                        ((0, 1), (2, 3), (4,), (5,)),
                                        (east, -east, north, -north))
        
        weights[S:S*S//2:S] = self._create_weights(v_cart,
                                        indices[S:S*S//2:S],
                                        neighbours_seam_south_east,
                                        ((0, 1), (2, 3), (4,), (5,)),
                                        (east, -east, north, -north))
        
        weights[0] = self._create_weights(v_cart,
                                        torch.asarray((0,), device=device),
                                        torch.asarray((V-S//2-1, S+1, V-S//2, 1, S), device=device),
                                        ((0,), (1,), (2, 3), (4, 4)),
                                        (east, -east, north, -north))
        
        weights[S//2] = self._create_weights(v_cart,
                                        torch.asarray((S//2,), device=device),
                                        self._basic_neighbours(V, indices[S//2].unsqueeze(0), (V-S//2-1, S+1, 1, V-1, S)),
                                        ((0,), (1,), (2, 2), (3, 4)),
                                        (east, -east, north, -north))

        return weights.contiguous()

_GLOBAL_WEIGHT_TABLES = NeighbourWeightTableCache()

@torch.jit.script
def _wrap_prev_diff(input_full, input, j, i, weights, output):
    output[:, 1:, ...] = (input_full[:, :4, j, i] - input[:, 1:, ...]) * weights
    output[:, 0, ...] = (input_full[:, 4, j, i] - input[:, 0, ...]) * weights

@torch.jit.script
def _wrap_prev_diff_acc(input_full, input, j, i, weights, output):
    output[:, 1:, ...] += (input_full[:, :4, j, i] - input[:, 1:, ...]) * weights
    output[:, 0, ...] += (input_full[:, 4, j, i] - input[:, 0, ...]) * weights

@torch.jit.script
def _wrap_next_diff(input_full, input, j, i, weights, output):
    output[:, :4, ...] = (input_full[:, 1:, j, i] - input[:, :4, ...]) * weights
    output[:, 5, ...] = (input_full[:, 0, j, i] - input[:, 4, ...]) * weights

@torch.jit.script
def _wrap_next_diff_acc(input_full, input, j, i, weights, output):
    output[:, :4, ...] += (input_full[:, 1:, j, i] - input[:, :4, ...]) * weights
    output[:, 5, ...] += (input_full[:, 0, j, i] - input[:, 4, ...]) * weights

@torch.jit.script
def _get_gradient_impl(L, input, poles, weight_table, output):
    HS, S = utils.get_face_side_dims(L)
    # input [C, face_side, width, height]
    # output [C, dir, face_side, width, height]    

    # Inner
    inner_in = input[..., 1:-1, 1:-1]
    inner_out = output[..., 1:-1, 1:-1]
    inner_out[:, 0, ...] = (input[:, :, :-2, :-2] - inner_in) * weight_table.inner[None, None, 0, ...] #east
    inner_out[:, 1, ...] = (input[:, :, 2:, 2:] - inner_in) * weight_table.inner[None, None, 1, ...] # west
    inner_out[:, 2, ...] = (input[:, :, :2, 1:-1] - inner_in) * weight_table.inner[None, None, 2, ...]   # north
    inner_out[:, 2, ...] += (input[:, :, 1:-1, 2:] - inner_in) * weight_table.inner[None, None, 3, ...]
    inner_out[:, 3, ...] = (input[:, :, 1:-1, :-2] - inner_in) * weight_table.inner[None, None, 4, ...]  # east
    inner_out[:, 3, ...] += (input[:, :, 2:, 1:-1] - inner_in) * weight_table.inner[None, None, 5, ...]

    # east
    east_in = input[..., 0, 1:HS]
    east_out = output[..., 0, 1:HS]
    _wrap_prev_diff(input, east_in, HS-1, slice(HS,S-1), weight_table.east[None, None, 0, ...], east_out[:, 1, ...])
    east_out[:, 1, ...] = (input[:, :, 1, 2:HS+1] - east_in) * weight_table.east[None, None, 1, ...]
    _wrap_prev_diff(input, east_in, HS-1, slice(HS+1, S), weight_table.east[None, None, 2, ...], east_out[:, 2, ...])
    east_out[:, 2, ...] += (input[:, :, 0, 2:HS+1] - east_in) * weight_table.east[None, None, 3, ...]
    east_out[:, 3, ...] = (input[:, :, 0, 0:HS-1] - east_in) * weight_table.east[None, None, 4, ...]
    east_out[:, 3, ...] += (input[:, :, 1, 1:HS] - east_in) * weight_table.east[None, None, 5, ...]

    # corner north
    cn_in = input[..., 0, HS]
    cn_out = output[..., 0, HS]
    _wrap_prev_diff(input, cn_in, HS-1, S-1, weight_table.cn[None, None, 0], cn_out[:, 0, :])
    cn_out[:, 1, :] = (input[:, :, 1, HS+1] - cn_in) * weight_table.cn[None, None, 1]
    cn_out[:, 2, :] = (input[:, :, 0, HS+1] - cn_in) * weight_table.cn[None, None, 2]
    cn_out[:, 3, :] = (input[:, :, 0, HS-1] - cn_in) * weight_table.cn[None, None, 3]
    cn_out[:, 3, :] += (input[:, :, 1, HS] - cn_in) * weight_table.cn[None, None, 4]

    # ne
    ne_in = input[..., 0, HS+1:S-1]
    ne_out = input[..., 0, HS+1:S-1]
    _wrap_prev_diff(input, ne_in, slice(HS-2, 0, -1), S-1, weight_table.ne[None, None, 0, :], ne_out[:, 0, ...])
    _wrap_prev_diff_acc(input, ne_in, slice(HS-1, 1, -1), S-1, weight_table.ne[None, None, 1, :], ne_out[:, 0, ...])
    ne_out[:, 1, ...] = (input[:, :, 1, HS+2:S] - ne_in) * weight_table.ne[None, None, 2, :]
    ne_out[:, 1, ...] += (input[:, :, 1, HS+1:S-1] - ne_in) * weight_table.ne[None, None, 3, :]
    ne_out[:, 2, ...] = (input[:, :, 0, HS+2:S] - ne_in) * weight_table.ne[None, None, 4, :]
    ne_out[:, 3, ...] = (input[:, :, 0, HS:S-2] - ne_in) * weight_table.ne[None, None, 5, :]

    # nne
    nne_in = input[..., 0, S-1]
    nne_out = output[..., 0, S-1]
    _wrap_prev_diff(input, nne_in, 0, S-1, weight_table.nne[None, None, 0], nne_out[:, 0, :])
    _wrap_prev_diff_acc(input, nne_in, 1, S-1, weight_table.nne[None, None, 1], nne_out[:, 0, :])
    _wrap_next_diff(input, nne_in, 0, S-1, weight_table.nne[None, None, 2], nne_out[:, 1, :])
    nne_out[:, 1] = (input[:, :, 1, S-1] - nne_in) * weight_table.nne[None, None, 3]
    nne_out[:, 2] = (poles[:, 1] - nne_in) * weight_table.nne[None, None, 4]
    nne_out[:, 3] = (input[:, :, 0, S-2] - nne_in) * weight_table.nne[None, None, 5]

    # nw
    nw_in = input[..., 1:HS-1, S-1]
    nw_out = output[..., 1:HS-1, S-1]
    nw_out[:, 0, ...] = (input[:, :, 0:HS-2, S-2] - nw_in) * weight_table.nw[None, None, 0, :]
    _wrap_next_diff(input, nw_in, 0, slice(S-2,HS,-1), weight_table.nw[None, None, 1, :], nw_out[:, 1, ...])
    nw_out[:, 2, ...] = (input[:, :, 0:HS-2, S-1] - nw_in) * weight_table.nw[None, None, 2, :]
    _wrap_next_diff_acc(input, nw_in, 0, slice(S-1,HS+1,-1), weight_table.nw[None, None, 3, :], nw_out[:, 2, ...])
    nw_out[:, 3, ...] = (input[:, :, 1:HS-1, S-2] - nw_in) * weight_table.nw[None, None, 4, :]
    nw_out[:, 3, ...] += (input[:, :, 2:HS, S-1] - nw_in) * weight_table.nw[None, None, 5, :]

    # nww
    nww_in = input[..., HS-1, S-1]
    nww_out = output[..., HS-1, S-1]
    nww_out[:, 0, :] = (input[:, :, HS-2, S-2] - nww_in) * weight_table.nww[None, None, 0]
    _wrap_next_diff(input, nww_in, 0, HS, weight_table.nww[None, None, 1], nww_out[:, 1, :])
    nww_out[:, 2, :] = (input[:, :, HS-2, S-1] - nww_in) * weight_table.nww[None, None, 2]
    _wrap_next_diff_acc(input, nww_in, 0, HS+1, weight_table.nww[None, None, 3], nww_out[:, 2, :])
    nww_out[:, 3, :] = (input[:, :, HS-1, S-2] - nww_in) * weight_table.nww[None, None, 4]
    _wrap_next_diff_acc(input, nww_in, 0, HS-1, weight_table.nww[None, None, 5], nww_out[:, 3, :])

    #west
    west_in = input[..., HS-1, HS:S-1]
    west_out = output[..., HS-1, HS:S-1]
    west_out[:, 0, ...] = (input[:, :, HS-2, HS-1:S-2] - west_in) * weight_table.west[None, None, 0, :]
    _wrap_next_diff(input, west_in, 0, slice(1, HS), weight_table.west[None, None, 1, :], west_out[:, 1, ...])
    west_out[:, 2, ...] = (input[:, :, HS-2, HS:S-1] - west_in) * weight_table.west[None, None, 2, :]
    west_out[:, 2, ...] += (input[:, :, HS-1, HS+1:S] - west_in) * weight_table.west[None, None, 3, :]
    west_out[:, 3, ...] = (input[:, :, HS-1, HS-1:S-2] - west_in) * weight_table.west[None, None, 4, :]
    _wrap_next_diff_acc(input, west_in, 0, slice(0, HS-1), weight_table.west[None, None, 5, :], west_out[:, 3, ...])



class IcosphereGradientDirectional(nn.Module):
    def __init__(self, keep_original_value = False):
        super().__init__()

        self.keep_original_value = keep_original_value

    def forward(self, x):
        assert(len(x.shape) == 3)
        N, C, V = x.shape
        L = utils.get_icosphere_level(V)
        w, h = utils.get_face_side_dims(L)

        non_polar = x[..., :-2].view(N*C, 5, w, h)
        poles = x[..., 2:].view(N*C, 2)

        weight_table = _GLOBAL_WEIGHT_TABLES.get_table(L, x.device)

        if self.keep_original_value:
            output = torch.empty((N, C, 5, V), device=x.device, dtype=x.dtype)
            output[:, :, :, 4] = x
            out_view = output.view(N*C, 5, 5, w, h)
        else:
            output = torch.empty((N, C, 4, V), device=x.device, dtype=x.dtype)
            out_view = output.view(N*C, 4, 5, w, h)

        _get_gradient_impl(L, non_polar, poles, weight_table, out_view)

        return output   