from typing import Any

import yaml

from src.training.trainer import Trainer
from src.utils.paths import (EXPERIMENTS_KADID10K_PATH,
                             EXPERIMENTS_TID2008_PATH,
                             EXPERIMENTS_TID2013_PATH,
                             EXPERIMENTS_LIVE_PATH)


def main() -> None:
    print('Ładuję datasety do szkolenia modelu ViT... Proszę poczekaj chwilę...\n')

    trainer = Trainer(experiment_path=(EXPERIMENTS_LIVE_PATH / 'test'))
    trainer.train()


if __name__ == '__main__':
    main()
