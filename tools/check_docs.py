"""Validate local documentation files, heading anchors and optional built HTML."""
from pathlib import Path
import argparse
import sys

from build_docs import ROOT, Links, local_target, pages, render, target_path


def check(root: Path, site: Path | None = None) -> list[str]:
    root = root.resolve()
    documents = pages(root)
    parsed = {}
    for page in documents:
        links = Links();links.feed(render((root/page).read_text(encoding='utf-8')))
        parsed[page] = links
    errors = []
    for page, links in parsed.items():
        for href in links.links:
            try:
                target = local_target(root, page, href)
            except ValueError as exc:
                errors.append(str(exc));continue
            if target is None:
                continue
            relative, fragment = target
            if not (root/relative).is_file():
                errors.append(f'{page}: missing local file: {href}')
            elif fragment and relative.suffix == '.md':
                if relative not in parsed or fragment not in parsed[relative].ids:
                    errors.append(f'{page}: missing heading fragment: {href}')
    if site is not None:
        site = site.resolve()
        for page in documents:
            output = target_path(page)
            if not (site/output).is_file():
                errors.append(f'HTML page missing: {output}');continue
            links = Links();links.feed((site/output).read_text(encoding='utf-8'))
            for href in links.links:
                try:target = local_target(site, output, href)
                except ValueError as exc:
                    errors.append(str(exc));continue
                if target is None:continue
                relative, fragment = target
                if not (site/relative).is_file():
                    errors.append(f'{output}: missing HTML asset/page: {href}')
                elif fragment and relative.suffix == '.html':
                    other = Links();other.feed((site/relative).read_text(encoding='utf-8'))
                    if fragment not in other.ids:
                        errors.append(f'{output}: missing HTML heading: {href}')
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--site', type=Path)
    args = parser.parse_args()
    errors = check(args.root, args.site)
    if errors:
        print('\n'.join(errors), file=sys.stderr)
        raise SystemExit(1)
    print(f'Documentation file links and heading anchors are valid ({len(pages(args.root))} pages).')


if __name__ == '__main__':
    main()
