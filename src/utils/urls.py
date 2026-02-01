from pathlib import Path, PurePosixPath
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse, unquote
import shutil


def download_file(
        file_url: str,
        destination_file_path: Path,
        overwrite: bool = False,
        timeout_seconds: int = 60
) -> Path:
    destination_file_path.parent.mkdir(parents=True, exist_ok=True)

    if not overwrite and destination_file_path.exists() and destination_file_path.is_file():
        raise FileExistsError(
            f"Error: Pobierany plik istnieje już we wskazanej lokalizacji!\n"
            f"Jeśli chcesz go nadpisać, dodaj parametr `overwrite=True`.\n"
            f"Ścieżka: {destination_file_path}"
        )

    request = Request(
        url=file_url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; dataset-downloader/1.0)"},
    )

    try:
        with urlopen(request, timeout=timeout_seconds) as response_stream:
            # Otwarcie docelowego pliku w trybie zapisu binarnego
            with open(destination_file_path, "wb") as output_file:
                # Kopiowanie zwróconych bajtów do docelowego pliku poprzez buforowanie / streamowanie
                shutil.copyfileobj(response_stream, output_file)  # type: ignore[arg-type]

    except HTTPError as http_error:
        # HTTPError zawiera kody statusów i wiadomości HTTP (np. 404, 403)
        raise RuntimeError(
            f"Error: Błąd HTTP w trakcie pobierania pliku!\n"
            f"    status={http_error.code}\n"
            f"    reason={http_error.reason}\n"
            f"    url={file_url}"
        ) from http_error

    except URLError as url_error:
        # URLError obejmuje problemy z DNS, odmowę połączenia, brak dostępu do Internetu itp.
        raise RuntimeError(
            f"Error: Błąd Sieci albo URL w trakcie pobierania pliku!\n"
            f"    url={file_url}\n"
            f"    error={url_error}"
        ) from url_error

    return destination_file_path


def get_file_name(url: str) -> str:
    parsed_url = urlparse(url)
    decoded_path = unquote(parsed_url.path)
    file_name = PurePosixPath(decoded_path).name

    return file_name


DATASET_TID2008_URL = 'http://www.ponomarenko.info/tid/tid2008.rar'
DATASET_TID2013_URL = 'http://www.ponomarenko.info/tid2013/tid2013.rar'
DATASET_KADID10K_URL = 'https://datasets.vqa.mmsp-kn.de/archives/kadid10k.zip'
