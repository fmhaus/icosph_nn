import torch
from icosph_nn.icosphere import IcoSphere, cart2sph
from collections import defaultdict

def get_icosphere_level(length):
    reduced = length - 2
    d, m = divmod(reduced, 10)
    assert(m == 0 and d > 0 and (d & (d-1)) == 0 and (d & 0x55555555) != 0, f"Invalid sequence length: {length}")
    return d.bit_length() // 2 - 1


class NeighbourWeightTables:
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
    
w = NeighbourWeightTables()
w.get_table(5)