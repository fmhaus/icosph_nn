import torch
from icosph_nn.icosphere import Icosphere, cart2sph
from collections import defaultdict

def get_icosphere_level(length):
    reduced = length - 2
    d, m = divmod(reduced, 10)
    assert m == 0
    assert d > 0 and (d & (d-1)) == 0 and (d & 0x55555555) != 0, f"Invalid sequence length: {length}"
    return (d.bit_length() - 1) // 2

def get_face_side_dims(level):
    two_pow_l = 2**level
    return (two_pow_l, 2*two_pow_l)