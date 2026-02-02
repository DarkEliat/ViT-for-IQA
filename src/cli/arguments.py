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
        help='Względna ścieżka do folderu konkretnego eksperymentu, znajdującego się w folderze `ViT-for-IQA/experiments/`, gdzie `ViT-for-IQA` to root projektu tej aplikacji.',
    )


def add_checkpoint_path_arg(parser: argparse.ArgumentParser, *, required: bool) -> None:
    parser.add_argument(
        '--checkpoint-path',
        required=required,
        help='Względna ścieżka do pliku checkpointu (.pth) konkretnego eksperymentu, znajdującego się w folderze `ViT-for-IQA/experiments/`, gdzie `ViT-for-IQA` to root projektu tej aplikacji.',
    )


def add_training_config_name_arg(parser: argparse.ArgumentParser, *, required: bool = True) -> None:
    parser.add_argument(
        '--training-config-name',
        required=required,
        help='Plik konfiguracyjny YAML z `configs/` określający szczegóły treningu.',
    )


def add_dataset_name_arg(parser: argparse.ArgumentParser, *, required: bool = True) -> None:
    parser.add_argument(
        '--dataset-name',
        required=required,
        help='Nazwa jednej z używanych w projekcie baz danych (obrazów).',
    )


def add_split_name_arg(parser: argparse.ArgumentParser, *, required: bool = True) -> None:
    parser.add_argument(
        '--split-name',
        required=required,
        help='Nazwa splitu datasetu. Dostępne splity: `train`, `val`, `test`.',
    )



