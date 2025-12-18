import math
import torch

def sph2cart(coords, radians = True):
    coords = torch.as_tensor(coords, dtype=torch.float32)
    lon, lat = coords[..., 0], coords[..., 1]
    if not radians:
        lon = torch.deg2rad(lon)
        lat = torch.deg2rad(lat)

    cos_lat = torch.cos(lat)
    x = cos_lat * torch.cos(lon)
    y = torch.sin(lat)
    z = cos_lat * torch.sin(lon)
    return torch.stack((x, y, z), dim=-1)

def cart2sph(coords, radians = True):
    coords = torch.as_tensor(coords, dtype=torch.float32)
    x, y, z = coords[..., 0], coords[..., 1], coords[..., 2]
    lon = torch.atan2(z, x)
    lat = torch.asin(y)
    if not radians:
        lon = torch.rad2deg(lon)
        lat = torch.rad2deg(lat)
    return torch.stack((lon, lat), dim=-1)

class Icosphere:
    def __init__(self, level, device='cpu'):
        self.level = level
        self.device = device

        # construct the level 0 icosahedron

        # latitude angle for the 2 horizontal rings (+-) 
        lat = math.degrees(math.atan(0.5))

        # functions for generating vertices
        def v_north():
            return sph2cart((0, 90), radians=False).to(device)
        
        def v_south():
            return sph2cart((0, -90), radians=False).to(device)
        
        def v_low(i):
            return sph2cart((72 * i, -lat), radians=False).to(device)
        
        def v_high(i):
            return sph2cart((72 * i + 36, lat), radians=False).to(device)

        # construct face begin points for the 20 faces using spherical coords
        self.face_begins = torch.concat([torch.stack((
            v_low(i),
            v_low(i),
            v_high(i),
            v_high(i)
        )) for i in range(5)])
        # construct face directions
        self.dir_up = torch.concat([torch.stack((
            v_low(i+1) - v_south(),
            v_high(i) - v_low(i),
            v_high(i+1) - v_low(i+1),
            v_north() - v_high(i),
        )) for i in range(5)])
        self.dir_side = torch.concat([torch.stack((
            v_south() - v_low(i),
            v_low(i+1) - v_low(i),
            v_low(i+1) - v_high(i),
            v_high(i+1) - v_high(i),
        )) for i in range(5)])

        self.face_north = v_north()
        self.face_south = v_south()

        self.vertex_count = 10 * (4 ** self.level) + 2
        self.vertex_north_index = self.vertex_count - 1
        self.vertex_south_index = self.vertex_count - 2

    def get_vertex_count(self):
        return self.vertex_count

    def generate_vertices(self, cartesian = True, radians = True):
        # vertices are construct using half rings from south to north pole (every reference to "ring" actually means half ring)
        n_rings_face = 2 ** self.level  # number of rings across a face
        ring_size = 2 * n_rings_face    # vertices per ring
        n_rings = 5 * n_rings_face      # total number of rings around the sphere

        # calculate face indices for face pair
        lower_face_indices = torch.triu(torch.ones((n_rings_face, n_rings_face), dtype=torch.int32, device=self.device))
        # repeat pattern to northen half of sphere
        face_indices = torch.concat((lower_face_indices, lower_face_indices + 2), dim=1)
        # repeat for 5 sides of the icosahedron
        full_face_indices = 4 * torch.arange(5, device=self.device)[:, None, None] + face_indices[None, :, :]

        # get barycentric coords towards triangle side
        parity = torch.arange(n_rings, device=self.device) % n_rings_face
        bary_side = parity / n_rings_face
        bary_side = bary_side.reshape((5, n_rings_face))[:, :, None]

        # get barycentric coords towards triangle up
        ring_grid_indices = torch.arange(n_rings_face, device=self.device)
        gi, gj = ring_grid_indices.unsqueeze(1), ring_grid_indices.unsqueeze(0)
        up_grid = torch.where(gi > gj, gj, gj-gi)
        up_grid = up_grid / n_rings_face
        bary_up = torch.concat((up_grid, up_grid), dim=1)[None, :, :]

        # combine indices
        off_up = bary_up[:, :, :, None] * self.dir_up[full_face_indices]
        off_side = bary_side[:, :, :, None] * self.dir_side[full_face_indices]
        ring_vertices = self.face_begins[full_face_indices] + off_up + off_side

        # normalize
        vec_lengths = torch.norm(ring_vertices, dim=3, keepdim=True)
        ring_vertices /= vec_lengths

        # add poles
        vertices_cart = torch.concat((ring_vertices.reshape(ring_size * n_rings, 3), 
                                      self.face_south[None, :], 
                                      self.face_north[None, :]))

        if cartesian:
            return vertices_cart

        # convert to spherical coords
        x, y, z = vertices_cart[:,0], vertices_cart[:,1], vertices_cart[:,2]
        lat = torch.arccos(y)
        lon = torch.arctan2(z, x)
        if not radians:
            lat = torch.rad2deg(lat)
            lon = torch.rad2deg(lon)
        
        return torch.stack([lon, lat], dim=1)

    def generate_triangles(self):
        n_rings_face = 2 ** self.level
        ring_size = 2 * n_rings_face
        n_rings = 5 * n_rings_face

        # triangles indices for a single half of a half ring
        trig_row1 = torch.arange(ring_size, device=self.device).repeat_interleave(2, dim=0)
        trig_pattern = torch.as_tensor((ring_size + 1, 1), device=self.device)
        trig_row2 = trig_row1 + torch.tile(trig_pattern, (ring_size,))
        trig_row3 = torch.concat((torch.as_tensor([ring_size], device=self.device), trig_row1[:-1] + ring_size + 1))
        ring_trig = torch.stack((trig_row1, trig_row2, trig_row3), dim = 1)

        # repeat ring for a whole face
        ring_trig = ring_trig[None, :, :].repeat(n_rings_face, 1, 1)
        ring_offset = torch.arange(n_rings_face, device=self.device) 
        ring_trig += ring_offset[:, None, None] * ring_size

        # adjust last half ring per face for face seam (side)
        ring_trig[-1, :, 2] -= n_rings_face
        ring_trig[-1, :, 1] -= torch.tile(torch.as_tensor([n_rings_face, 0], device=self.device), (ring_size,))

        # seam connection north; connect to last vertices in half ring to next pole triangle
        seam_north = torch.arange(ring_size * (1 + n_rings_face),
                                   (1 + ring_size) * n_rings_face - 1,
                                   -1,
                                   device=self.device)
        seam_north = torch.stack((seam_north[1:], seam_north[:-1]), dim=1)
        ring_trig[:, 2 * ring_size - 2, 1] = seam_north[:, 0]
        ring_trig[:, 2 * ring_size - 1, 1] = seam_north[:, 1]
        ring_trig[:, 2 * ring_size - 1, 2] = seam_north[:, 0]

        # conect first vertices from last half ring in face to next lower pole triangle
        seam_south = torch.arange(2 * n_rings_face * ring_size, 
                                  n_rings_face * ring_size - 1, 
                                  -ring_size, 
                                  device=self.device)
        ring_trig[-1, 0:2*n_rings_face, 2] = seam_south.repeat_interleave(2)[1:-1]
        ring_trig[-1, 0:2*n_rings_face:2, 1] = seam_south[1:]

        # repeat faces 5 times around full sphere with offsets
        ring_offsets = torch.arange(5, device=self.device) * (ring_size * n_rings_face)
        full_ring_trigs = ring_offsets[:, None, None, None] + ring_trig[None, :, :, :]
        # mod last indices
        full_ring_trigs[-1] = full_ring_trigs[-1] % (ring_size * n_rings)
        # set pole vertex connections
        full_ring_trigs[:, -1, 0, 2] = self.vertex_south_index
        full_ring_trigs[:, 0, -1, 1] = self.vertex_north_index

        return full_ring_trigs.reshape(2 * ring_size * n_rings, 3)

    def get_east_directions(v_sph):
        assert(len(v_sph.shape) == 2)
        assert(v_sph.shape[1] == 2)

        lon = v_sph[:, 0]
        return torch.stack((-torch.sin(lon), 
                            torch.zeros(v_sph.shape[0], device=v_sph.device), 
                            torch.cos(lon)), 
                            dim=1)
    
    def get_north_directions(v_sph):
        assert(len(v_sph.shape) == 2)
        assert(v_sph.shape[1] == 2)

        lon, lat = v_sph[:, 0], v_sph[:, 1]
        sin_lat = torch.sin(lat)
        return torch.stack((-sin_lat * torch.cos(lon), 
                            torch.cos(lat),
                            -sin_lat * torch.sin(lon)), 
                            dim=1)
    