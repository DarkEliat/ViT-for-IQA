from torch.utils.data import Dataset as TorchDataset

from src.datasets.base_dataset import BaseDataset
from src.datasets.kadid_dataset import Kadid10kDataset
from src.datasets.live_dataset import LiveDataset
from src.datasets.tid_dataset import Tid2008Dataset, Tid2013Dataset
from src.utils.data_types import Config


def build_dataset(config: Config) -> TorchDataset | BaseDataset:
    match config['dataset']['name']:
        case 'kadid10k':
            return Kadid10kDataset(config=config)
        case 'tid2008':
            return Tid2008Dataset(config=config)
        case 'tid2013':
            return Tid2013Dataset(config=config)
        case 'live':
            return LiveDataset(config=config)
        case _:
            raise ValueError(f"Error: Niewspierany typ bazy danych: `{config['dataset']['name']}`!")
