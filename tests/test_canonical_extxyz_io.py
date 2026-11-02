from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


HAS_ASE = importlib.util.find_spec("ase") is not None


@unittest.skipUnless(HAS_ASE, "ase is required for canonical extxyz IO tests")
class CanonicalExtxyzIOTests(unittest.TestCase):
    def _atoms(self, x: float):
        from ase import Atoms

        atoms = Atoms("H", positions=[[x, 0.0, 0.0]])
        atoms.info["structure_id"] = f"h-{x}"
        return atoms

    def test_concat_extxyz_files_preserves_multiframe_order(self) -> None:
        from workflow_core.structures.formats import concat_extxyz_files, read_extxyz_structures, write_extxyz_structures

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = write_extxyz_structures(root / "a.xyz", [self._atoms(0.0), self._atoms(1.0)])
            b = write_extxyz_structures(root / "b.xyz", [self._atoms(2.0)])

            target = concat_extxyz_files([a, b], root / "merged.xyz")

            frames = read_extxyz_structures(target, allow_empty=False)
            self.assertEqual([frame.info["structure_id"] for frame in frames], ["h-0.0", "h-1.0", "h-2.0"])

    def test_empty_extxyz_policy_is_explicit(self) -> None:
        from workflow_core.structures.formats import read_extxyz_structures, validate_extxyz_file, write_extxyz_structures

        with tempfile.TemporaryDirectory() as tmp:
            empty = write_extxyz_structures(Path(tmp) / "empty.xyz", [], allow_empty=True)

            self.assertEqual(read_extxyz_structures(empty, allow_empty=True), [])
            self.assertTrue(validate_extxyz_file(empty, allow_empty=True))
            with self.assertRaisesRegex(ValueError, "empty extxyz is not allowed"):
                read_extxyz_structures(empty, allow_empty=False)
            self.assertFalse(validate_extxyz_file(empty, allow_empty=False))

    def test_invalid_extxyz_fails_validation(self) -> None:
        from workflow_core.structures.formats import read_extxyz_structures, validate_extxyz_file

        with tempfile.TemporaryDirectory() as tmp:
            invalid = Path(tmp) / "invalid.xyz"
            invalid.write_text("this is not extxyz\n", encoding="utf-8")

            self.assertFalse(validate_extxyz_file(invalid, allow_empty=False))
            with self.assertRaisesRegex(ValueError, "invalid extxyz"):
                read_extxyz_structures(invalid, allow_empty=False)

    def test_concat_rejects_empty_source_when_disallowed(self) -> None:
        from workflow_core.structures.formats import concat_extxyz_files, write_extxyz_structures

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            empty = write_extxyz_structures(root / "empty.xyz", [], allow_empty=True)
            nonempty = write_extxyz_structures(root / "nonempty.xyz", [self._atoms(0.0)])

            with self.assertRaisesRegex(ValueError, "empty extxyz is not allowed"):
                concat_extxyz_files([empty, nonempty], root / "merged.xyz", allow_empty_sources=False)


if __name__ == "__main__":
    unittest.main()
