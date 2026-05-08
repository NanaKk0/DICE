from pathlib import Path
from ase.io import read, write


def collect_user_xyz(dataset_dir="user_assets"):
    """
    扫描 user_assets 目录下所有 xyz 文件
    """
    dataset_dir = Path(dataset_dir)

    if not dataset_dir.exists():
        return []

    xyz_files = list(dataset_dir.rglob("*.xyz"))

    return xyz_files


def merge_xyz_files(xyz_files, output_file):
    """
    合并多个 xyz
    """

    all_atoms = []

    for f in xyz_files:

        try:
            atoms_list = read(f, ":")
            all_atoms.extend(atoms_list)

        except Exception as e:

            print(f"[WARN] Failed to read {f}: {e}")

    if len(all_atoms) == 0:
        return False

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    write(output_file, all_atoms)

    return True
