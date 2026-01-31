import argparse


def add_experiment_name_arg(parser: argparse.ArgumentParser, *, required: bool = True) -> None:
    parser.add_argument(
        '--experiment-name',
        required=required,
        help='Nazwa nowego eksperymentu.',
    )


def add_experiment_path_arg(parser: argparse.ArgumentParser, *, required: bool = True) -> None:
    parser.add_argument(
        '--experiment-path',
        required=required,
        help='Ścieżka do folderu eksperymentu, np. `dataset_name/experiment_name/`.',
    )


def add_checkpoint_name_arg(parser: argparse.ArgumentParser, *, required: bool) -> None:
    parser.add_argument(
        '--checkpoint-name',
        required=required,
        help='Nazwa pliku checkpointu (.pth) wewnątrz folderu `<experiment_path>/checkpoints/`, np. `best.pth`.',
    )


def add_split_name_arg(parser: argparse.ArgumentParser, *, required: bool = True) -> None:
    parser.add_argument(
        '--split-name',
        required=required,
        help='Nazwa splitu datasetu. Dostępne splity: `train`, `val`, `test`.',
    )


def add_global_config_name_arg(parser: argparse.ArgumentParser, *, required: bool = True) -> None:
    parser.add_argument(
        '--global-config-name',
        required=required,
        help='Plik konfiguracyjny YAML z `configs/`.',
    )
