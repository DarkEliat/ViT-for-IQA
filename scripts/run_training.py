from typing import Any

import yaml

from src.training.trainer import Trainer
from src.utils.paths import CONFIGS_PATH


def main() -> None:
    config_file_path = CONFIGS_PATH / 'train_kadid10k_baseline.yaml'

    with open(config_file_path, 'r') as config_file:
        config: dict[str, Any] = yaml.safe_load(config_file)

    trainer = Trainer(config=config)
    trainer.train()


if __name__ == '__main__':
    main()
