from pathlib import Path
from pprint import pprint

from src.inference.predictor import Predictor
from src.utils.paths import (EXPERIMENTS_KADID10K_PATH,
                             EXPERIMENTS_TID2008_PATH,
                             EXPERIMENTS_TID2013_PATH,
                             EXPERIMENTS_LIVE_PATH)


def main() -> None:
    predictor = Predictor(
        training_config_path=(EXPERIMENTS_TID2008_PATH / 'test' / 'config.yaml'),
        checkpoint_path=(EXPERIMENTS_TID2008_PATH / 'test' / 'checkpoints' / 'last.pth')
    )

    predicted_quality_scores = predictor.predict_on_training_dataset()

    print()
    pprint(predicted_quality_scores)


if __name__ == '__main__':
    main()
