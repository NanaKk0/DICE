"""Check user-visible navigation/rendering and link failures."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_rendered_site_keeps_steps_code_and_heading_links(tmp_path):
    subprocess.run([sys.executable, str(ROOT/'tools/build_docs.py'), '--output', str(tmp_path/'site')], check=True)
    from html.parser import HTMLParser
    class Headings(HTMLParser):
        ids = set()
        def handle_starttag(self, tag, attrs):
            values = dict(attrs)
            if tag in ('h1','h2','h3') and 'id' in values:
                self.ids.add(values['id'])
    page = (tmp_path/'site/docs/install.html').read_text()
    parsed = Headings();parsed.feed(page)
    assert 'obtain-and-install-the-control-package' in parsed.ids
    assert '<pre><code' in page
    assert '<ol>' in (tmp_path/'site/docs/first-principles.html').read_text()
    assert (tmp_path/'site/LICENSING.html').is_file()
    assert (tmp_path/'site/docs/index.html').is_file()
    subprocess.run([sys.executable, str(ROOT/'tools/check_docs.py'), '--site', str(tmp_path/'site')], check=True)


def test_link_checker_detects_missing_files_and_heading_fragments(tmp_path):
    root=tmp_path/'checkout'
    (root/'docs').mkdir(parents=True)
    (root/'README.md').write_text('# Home\n\n[Guide](docs/guide.md#details)\n')
    (root/'docs/guide.md').write_text('# Guide\n\n## Details\n\n1. First\n2. Second\n')
    command=[sys.executable,str(ROOT/'tools/check_docs.py'),'--root',str(root)]
    assert subprocess.run(command,capture_output=True).returncode == 0
    (root/'docs/guide.md').write_text('# Guide\n\n[Missing](absent.md)\n')
    failed=subprocess.run(command,capture_output=True,text=True)
    assert failed.returncode != 0
    assert 'details' in failed.stdout + failed.stderr
    assert 'absent.md' in failed.stdout + failed.stderr
