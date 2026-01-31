from src.cli.entrypoint import run_cli_command
from src.cli.create_experiment_cli import ExperimentCreationCliCommand


if __name__ == '__main__':
    run_cli_command(ExperimentCreationCliCommand())
