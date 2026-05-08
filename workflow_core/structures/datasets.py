from pathlib import Path

from ase.io import read, write


def collect_xyz_files(folder):
    folder = Path(folder)

    if not folder.exists():
        return []

    return list(folder.rglob("*.xyz"))


def merge_xyz_files(files, output_file):
    atoms_all = []

    for file_path in files:
        try:
            atoms_all.extend(read(file_path, ":"))
        except Exception as exc:
            print(f"[WARN] Failed to read {file_path}: {exc}")

    if len(atoms_all) == 0:
        raise RuntimeError("No valid xyz structures found.")

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    write(output_file, atoms_all)

    return output_file
