import argparse
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar


TNormalizedArguments = TypeVar("TNormalizedArguments")


@dataclass(frozen=True)
class CliExecutionResult:
    exit_code: int


class BaseCliCommand(ABC, Generic[TNormalizedArguments]):
    def __init__(self) -> None:
        self._arg_parser = self._build_arg_parser()


    @property
    @abstractmethod
    def command_name(self) -> str:
        raise NotImplementedError


    @property
    @abstractmethod
    def command_description(self) -> str:
        raise NotImplementedError


    def _build_arg_parser(self) -> argparse.ArgumentParser:
        arg_parser = argparse.ArgumentParser(
            prog=self.command_name,
            description=self.command_description,
        )

        self.add_args(parser=arg_parser)

        return arg_parser


    @abstractmethod
    def add_args(self, parser: argparse.ArgumentParser) -> None:
        raise NotImplementedError


    @abstractmethod
    def validate_and_normalize_args(self, parsed_namespace: argparse.Namespace) -> TNormalizedArguments:
        raise NotImplementedError


    @abstractmethod
    def run_command(self, normalized_args: TNormalizedArguments) -> None:
        raise NotImplementedError


    def execute(self) -> CliExecutionResult:
        try:
            parsed_namespace = self._arg_parser.parse_args()
            normalized_arguments = self.validate_and_normalize_args(parsed_namespace=parsed_namespace)
            self.run_command(normalized_args=normalized_arguments)

            return CliExecutionResult(exit_code=0)

        except (ValueError, FileNotFoundError, NotADirectoryError) as exception:
            _print_user_error(exception)

            return CliExecutionResult(exit_code=2)

        except KeyboardInterrupt:
            print("\nDziałanie programu zostało przerwane przez użytkownika (Ctrl+C).", file=sys.stderr)

            return CliExecutionResult(exit_code=130)

        except Exception:
            print(f"\nError: Nieoczekiwany błąd w `{self.command_name}`:", file=sys.stderr)

            raise


def _print_user_error(exception: Exception) -> None:
    print(f"\nError: {exception}", file=sys.stderr)
    print("\nWskazówka: Uruchom z `--help`, aby zobaczyć dostępne argumenty.", file=sys.stderr)
