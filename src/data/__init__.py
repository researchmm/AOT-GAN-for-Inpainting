from .dataset import TerrainDataset

from torch.utils.data import DataLoader


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def create_loader(args):
    dataset = TerrainDataset(
        "data/download/MDRS/data/*.tif",
        dataset_type="train",
        randomize=True,
        block_variance=1,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size // args.world_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    return sample_data(data_loader)