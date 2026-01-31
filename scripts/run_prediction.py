from src.cli.entrypoint import run_cli_command
from src.cli.run_prediction_cli import PredictionCliCommand


if __name__ == '__main__':
    run_cli_command(PredictionCliCommand())
