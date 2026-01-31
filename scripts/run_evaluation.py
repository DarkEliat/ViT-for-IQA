from src.cli.entrypoint import run_cli_command
from src.cli.run_evaluation_cli import EvaluationCliCommand


if __name__ == '__main__':
    run_cli_command(EvaluationCliCommand())
