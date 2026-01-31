from src.evaluation.evaluator import Evaluator
from src.utils.paths import (
    EXPERIMENTS_KADID10K_PATH,
    EXPERIMENTS_TID2008_PATH,
    EXPERIMENTS_TID2013_PATH,
    EXPERIMENTS_LIVE_PATH
)

def main() -> None:
    evaluator = Evaluator(
        experiment_path=(EXPERIMENTS_LIVE_PATH / 'test/'),
        split_name='test',
        checkpoint_name='last.pth'
    )

    evaluator.evaluate(save_outputs=True)


if __name__ == '__main__':
    main()
