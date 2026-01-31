from src.cli.base_cli_command import BaseCliCommand


def run_cli_command(cli_command: BaseCliCommand) -> None:
    execution_result = cli_command.execute()

    raise SystemExit(execution_result.exit_code)
