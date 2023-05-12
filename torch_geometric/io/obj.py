import torch

from torch_geometric.data import Data


def yield_file(in_file):
    with open(in_file) as f:
        buf = f.read()
    for b in buf.split('\n'):
        if b.startswith('v '):
            yield ['v', [float(x) for x in b.split(" ")[1:]]]
        elif b.startswith('f '):
            triangles = b.split(' ')[1:]
            # -1 as .obj is base 1 but the Data class expects base 0 indices
            yield ['f', [int(t.split("/")[0]) - 1 for t in triangles]]
        else:
            yield ['', ""]


def read_obj(in_file):
    vertices = []
    faces = []

    for k, v in yield_file(in_file):
        if k == 'f':
            faces.append(v)

        elif k == 'v':
            vertices.append(v)
    if not len(faces) or not len(vertices):
        return None

    pos = torch.tensor(vertices, dtype=torch.float)
    face = torch.tensor(faces, dtype=torch.long).t().contiguous()

    return Data(pos=pos, face=face)
