from torch_geometric.datasets import QM7b
from torch_geometric.graphgym.register import register_loader


@register_loader('example')
def load_dataset_example(format, name, dataset_dir):
    dataset_dir = f'{dataset_dir}/{name}'
    if format == 'PyG' and name == 'QM7b':
        return QM7b(dataset_dir)
