from icosph_nn import icosphere
from icosph_nn import visuals
from icosph_nn import utils
import torch

L = 6
vis = visuals.IcosphereVisualizer(L, title="GG")

ico = icosphere.Icosphere(L)
vert = ico.generate_vertices(cartesian=False)
colors = torch.zeros((ico.get_vertex_count(), 3), dtype=torch.float32)

HS, S = utils.get_face_side_dims(L)
indices = torch.arange(HS*S).view(HS, S)
indices_next = indices + (HS * S)
indices_prev = indices + (HS * S * 4)

indices_main = indices[1:-1, 1:-1]
neighbours_indices = [
    indices[:-2, :-2],
    indices[2:, 2:],
    indices[:-2, 1:-1],
    indices[1:-1, 2:],
    indices[1:-1, :-2],
    indices[2:, 1:-1]
]

to_neighbour = vert[neighbours_indices[0]] - vert[indices_main]
v = torch.where(to_neighbour[..., 1] > 0, 
                            torch.ones(to_neighbour.shape[:-1]), 
                            torch.zeros(to_neighbour.shape[:-1]))

colors[indices_main, :] = torch.stack((v, torch.zeros(v.shape), torch.zeros(v.shape)), dim=-1)

R = ico.get_vertex_count() - HS * S
colors[HS*S:] = torch.stack((torch.zeros(R), torch.ones(R), torch.ones(R)), dim=-1)

vis.update_mesh(colors)
vis.main_loop()