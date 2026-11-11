"""Build the portable documentation with Python-Markdown and local assets."""
from __future__ import annotations

import argparse
import html
from html.parser import HTMLParser
import os
from pathlib import Path
import re
import shutil
from urllib.parse import urlsplit, unquote

import markdown

ROOT = Path(__file__).resolve().parents[1]


def pages(root: Path) -> list[Path]:
    candidates = [Path('README.md'), *[p.relative_to(root) for p in sorted((root/'docs').glob('*.md'))],
                  Path('examples/cpu_mock/README.md'), Path('examples/iteration_handoff/README.md'),
                  Path('LICENSING.md'), Path('LICENSE-DATA.md')]
    return [p for p in candidates if (root/p).is_file()]


def target_path(relative: Path) -> Path:
    return Path('index.html') if relative == Path('README.md') else relative.with_suffix('.html')


def render(text: str) -> str:
    return markdown.markdown(text, extensions=['fenced_code', 'tables', 'toc', 'sane_lists'], output_format='html')


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.ids = set()
    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if 'id' in values:
            self.ids.add(values['id'])
        key = 'href' if tag == 'a' else 'src' if tag == 'img' else None
        if key and key in values:
            self.links.append(values[key])


def local_target(root: Path, page: Path, href: str):
    parsed = urlsplit(html.unescape(href))
    if parsed.scheme or parsed.netloc:
        return None
    path = unquote(parsed.path)
    candidate = ((root/page).parent/path).resolve() if path else (root/page).resolve()
    if candidate != root.resolve() and root.resolve() not in candidate.parents:
        raise ValueError(f'{page}: link escapes documentation root: {href}')
    return candidate.relative_to(root.resolve()), unquote(parsed.fragment)


def rewrite_links(body: str, root: Path, page: Path) -> str:
    def replacement(match):
        href = html.unescape(match[2])
        resolved = local_target(root, page, href)
        if resolved is None:
            return match[0]
        relative, fragment = resolved
        if relative.suffix == '.md':
            destination = target_path(relative)
        else:
            destination = relative
        url = os.path.relpath(destination, target_path(page).parent)
        if fragment:
            url += '#' + fragment
        return match[1] + html.escape(url, quote=True) + match[3]
    return re.sub(r'((?:href|src)=")([^"]*)(")', replacement, body)


def navigation(root: Path, page: Path) -> str:
    groups = [('Start', [('Home', Path('README.md')), ('All guides', Path('docs/index.md'))])]
    directory = root/'docs/index.md'
    if directory.is_file():
        for line in directory.read_text().splitlines():
            if line.startswith('## '):
                groups.append((line[3:], []))
            for label, href in re.findall(r'\[([^\]]+)\]\(([^)]+)\)', line):
                target = local_target(root, Path('docs/index.md'), href)
                if target is not None and target[0].suffix == '.md':
                    groups[-1][1].append((label, target[0]))
    sections = []
    seen = set()
    for title, entries in groups:
        links = []
        for label, relative in entries:
            if relative in seen or not (root/relative).is_file():
                continue
            seen.add(relative)
            url = os.path.relpath(target_path(relative), target_path(page).parent)
            links.append(f'<a href="{html.escape(url, quote=True)}">{html.escape(label)}</a>')
        if links:
            sections.append('<section><strong>'+html.escape(title)+'</strong>'+''.join(links)+'</section>')
    return ''.join(sections)


def build(root: Path, destination: Path) -> int:
    root, destination = root.resolve(), destination.resolve()
    if destination == root or destination in root.parents:
        raise ValueError('output must not replace the documentation input root')
    destination.mkdir(parents=True, exist_ok=True)
    all_pages = pages(root)
    for relative in all_pages:
        text = (root/relative).read_text(encoding='utf-8')
        body = render(text)
        parser = Links();parser.feed(body)
        for href in parser.links:
            resolved = local_target(root, relative, href)
            if resolved is not None:
                asset = resolved[0]
                if asset.suffix != '.md' and (root/asset).is_file():
                    out = destination/asset;out.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(root/asset, out)
        title = text.splitlines()[0].lstrip('# ')
        document = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        document += '<title>'+html.escape(title)+' · DICE</title><style>'
        document += 'body{font:16px/1.65 system-ui,sans-serif;color:#172a3a;margin:0;display:grid;grid-template-columns:240px minmax(0,960px);gap:32px;padding:24px}nav section{margin-bottom:18px}nav a{display:block}a{color:#145b89}h1,h2,h3{line-height:1.25;scroll-margin-top:20px}pre{background:#edf2f6;padding:16px;overflow:auto}code{font-size:.9em}table{border-collapse:collapse;display:block;overflow:auto}td,th{border:1px solid #ccd4db;padding:8px;text-align:left}footer{border-top:1px solid #ccd4db;margin-top:30px}img{max-width:100%}@media(max-width:800px){body{display:block}nav{border-bottom:1px solid #ccd4db;margin-bottom:24px}}'
        document += '</style></head><body><nav aria-label="Documentation">'+navigation(root, relative)+'</nav><main>'+rewrite_links(body, root, relative)+'</main></body></html>'
        out = destination/target_path(relative);out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(document, encoding='utf-8')
    (destination/'.nojekyll').touch()
    print(f'Built {len(all_pages)} documentation pages in {destination}')
    return len(all_pages)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, default=ROOT/'_site')
    args = parser.parse_args()
    build(args.root, args.output)


if __name__ == '__main__':
    main()
