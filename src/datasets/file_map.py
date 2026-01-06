from pathlib import Path

from src.utils.paths import PROJECT_ROOT


class FileMap:
    def __init__(self, files_directory_path: Path) -> None:
        self.files_directory_path = files_directory_path
        self._file_map: dict[str, Path] = {}

        self.rebuild_map()


    @property
    def files_directory_path(self) -> Path:
        return self._files_directory_path

    @files_directory_path.setter
    def files_directory_path(self, new_directory_path: Path):
        new_directory_path = PROJECT_ROOT / new_directory_path

        if not new_directory_path or not new_directory_path.exists():
            raise FileNotFoundError(
                f"Error: Wskazany nowy folder z plikami do zmapowania nie istnieje!\n"
                f"Ścieżka: {new_directory_path}"
            )

        self._files_directory_path = new_directory_path

    @property
    def file_map(self) -> dict[str, Path]:
        return self._file_map


    def rebuild_map(self) -> None:
        if not self.files_directory_path or not self.files_directory_path.exists():
            raise FileNotFoundError(
                f"Error: Wskazany folder z plikami do zmapowania nie istnieje!\n"
                f"Ścieżka: {self.files_directory_path}"
            )

        self._file_map = {
            file_path.name.lower(): file_path
            for file_path in self.files_directory_path.iterdir()
            if file_path.is_file()
        }


    def get_file_path(self, file_name: str) -> Path:
        try:
            return self.file_map[file_name.lower()]
        except KeyError:
            raise FileNotFoundError(
                f"Error: Nie odnaleziono obrazu {file_name}!\n"
                f"Ścieżka: {self.files_directory_path}"
            )


    def has_file_path(self, file_name: str) -> bool:
        return file_name.lower() in self.file_map
