from typing import Any

import yaml

from src.training.trainer import Trainer
from src.utils.paths import EXPERIMENTS_KADID10K_PATH


def main() -> None:
    trainer = Trainer(experiment_path=(EXPERIMENTS_KADID10K_PATH / 'test'))
    trainer.train()


if __name__ == '__main__':
    main()
