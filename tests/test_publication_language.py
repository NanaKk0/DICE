"""Keep published text in English, including code, data headers and SVG assets."""
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
CJK = re.compile('[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U000323af]')
TEXT_SUFFIXES = {'.md', '.py', '.json', '.toml', '.yml', '.yaml', '.sh',
                 '.lammps', '.in', '.extxyz', '.xyz', '.svg', '.txt'}


def test_publication_text_contains_no_cjk_characters():
    directories = ['docs', 'examples', 'skills', 'tests', 'tools', '.github',
                   'workflow_app', 'workflow_artifacts', 'workflow_backends',
                   'workflow_core', 'workflow_policies', 'workflow_runtime']
    files = [p for p in ROOT.iterdir() if p.is_file()]
    for name in directories:
        for path in (ROOT / name).rglob('*'):
            if any(part in {'__pycache__', '.pytest_cache', '.dice_state'}
                   or re.fullmatch(r'iter\d+', part) for part in path.parts):
                continue
            if path.is_file() and path.suffix in TEXT_SUFFIXES:
                files.append(path)
    for path in files:
        if path.suffix in TEXT_SUFFIXES or path.name in {'LICENSE', '.gitignore'}:
            assert not CJK.search(path.read_text(encoding='utf-8')), path.relative_to(ROOT)
