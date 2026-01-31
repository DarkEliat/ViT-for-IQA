from src.cli.entrypoint import run_cli_command
from src.cli.run_training_cli import TrainingCliCommand

if __name__ == '__main__':
    run_cli_command(TrainingCliCommand())
