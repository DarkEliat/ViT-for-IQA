from typing import Any

import yaml

from src.training.trainer import Trainer
from src.utils.paths import CONFIG_TRAIN_KADID10K_BASELINE_PATH


def main() -> None:
    with open(CONFIG_TRAIN_KADID10K_BASELINE_PATH, 'r') as config_file:
        config: dict[str, Any] = yaml.safe_load(config_file)

    trainer = Trainer(config=config)
    trainer.train()


if __name__ == '__main__':
    main()
