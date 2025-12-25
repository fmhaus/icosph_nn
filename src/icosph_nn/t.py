from icosph_nn import icosphere
from icosph_nn import visuals
from icosph_nn import utils
import torch

L = 6
vis = visuals.IcosphereVisualizer(L, title="GG")

ico = icosphere.Icosphere(L)
vert = ico.generate_vertices(cartesian=False)
colors = torch.ones((ico.get_vertex_count(), 3), dtype=torch.float32) / 2

HS, S = utils.get_face_side_dims(L)
indices = torch.arange(HS*S).view(HS, S)
indices_next = indices + (HS * S)
indices_prev = indices + (HS * S * 4)

indices_main = indices[1:-1, 1:-1]
neighbours_indices = torch.stack([
    indices[:-2, :-2],
    indices[2:, 2:],
    indices[:-2, 1:-1],
    indices[1:-1, 2:],
    indices[1:-1, :-2],
    indices[2:, 1:-1]
], dim=0)

to_neighbour = vert[neighbours_indices] - vert[indices_main[None, :]]
to_neighbour[..., 0] = (to_neighbour[..., 0] + torch.pi) % (2 * torch.pi) - torch.pi
print(to_neighbour.shape)

dir = torch.asarray([0, 1], dtype=torch.float)
w = torch.sum(to_neighbour * dir[..., :], dim=-1) / (torch.sqrt(torch.sum(to_neighbour ** 2, dim=-1)))
v, i = torch.topk(w, k=2, dim=0)

print(i.shape)
COLORS = torch.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=torch.float32)

colors[indices_main, :] = COLORS[i[0]]

vis.update_mesh(colors)
vis.main_loop()