"""Download the pinned DPA4-Neo foundation used by this example."""
from pathlib import Path
from urllib.request import urlopen


REPOSITORY = "deepmodelingcommunity/DPA4-OMat24"
REVISION = "173bfd5a26f90a756ee891777270ef117078d9be"
FILENAME = "DPA4-Neo-OMat24-v20260805.pt"


def main():
    directory = Path(__file__).resolve().parent / "models"
    directory.mkdir(exist_ok=True)
    target = directory / FILENAME
    if target.is_file() and target.stat().st_size:
        print(f"Using {target}")
        return
    temporary = target.with_suffix(".download")
    url = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{FILENAME}"
    try:
        with urlopen(url, timeout=60) as response, temporary.open("wb") as output:
            while block := response.read(1024 * 1024):
                output.write(block)
        if temporary.stat().st_size == 0:
            raise ValueError("The downloaded checkpoint is empty")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Downloaded {target}")


if __name__ == "__main__":
    main()
