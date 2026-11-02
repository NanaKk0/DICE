"""Render the curated Markdown documentation with Python's standard library."""
from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Resolve the file list relative to the distribution, not the caller's cwd.
PAGES = [Path("README.md"), *[p.relative_to(ROOT) for p in sorted((ROOT / "docs").glob("*.md"))], Path("examples/cpu_mock/README.md"), Path("examples/iteration_handoff/README.md")]


def target_path(relative: Path) -> Path:
    return Path("index.html") if relative == Path("README.md") else relative.with_suffix(".html")


def inline(text: str) -> str:
    tokens: list[str] = []

    def stash(value: str) -> str:
        tokens.append(value)
        return f"\x00{len(tokens) - 1}\x00"

    text = re.sub(r"`([^`]+)`", lambda m: stash("<code>" + html.escape(m[1]) + "</code>"), text)

    def link(match: re.Match) -> str:
        href = match[2]
        if href.lower().startswith(("javascript:", "data:")):
            return html.escape(match[1])
        href = re.sub(r"\.md(?=($|#))", ".html", href)
        if href == "../README.html":
            href = "../index.html"
        return stash(f'<a href="{html.escape(href, quote=True)}">{html.escape(match[1])}</a>')

    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", link, text)
    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    return re.sub(r"\x00(\d+)\x00", lambda m: tokens[int(m[1])], text)


def render(markdown: str) -> str:
    output: list[str] = []
    paragraph: list[str] = []
    code: list[str] | None = None
    listing = False
    table = False

    def flush() -> None:
        if paragraph:
            output.append("<p>" + inline(" ".join(paragraph)) + "</p>")
            paragraph.clear()

    for line in markdown.splitlines():
        if line.startswith("```"):
            flush()
            if code is None:
                code = []
            else:
                output.append("<pre><code>" + html.escape("\n".join(code)) + "</code></pre>")
                code = None
            continue
        if code is not None:
            code.append(line)
            continue
        is_list = line.startswith("- ")
        is_table = line.startswith("|")
        if listing and not is_list:
            output.append("</ul>")
            listing = False
        if table and not is_table:
            output.append("</tbody></table></div>")
            table = False
        if not line.strip():
            flush()
        elif line.startswith("#"):
            flush()
            heading, title = line.split(" ", 1)
            level = min(len(heading), 6)
            output.append(f"<h{level}>" + inline(title) + f"</h{level}>")
        elif is_list:
            flush()
            if not listing:
                output.append("<ul>")
                listing = True
            output.append("<li>" + inline(line[2:]) + "</li>")
        elif is_table:
            flush()
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if all(re.fullmatch(r":?-+:?", cell) for cell in cells):
                continue
            if not table:
                output.append('<div class="table"><table><tbody>')
                table = True
            output.append("<tr>" + "".join("<td>" + inline(cell) + "</td>" for cell in cells) + "</tr>")
        else:
            paragraph.append(line.strip())
    flush()
    if listing:
        output.append("</ul>")
    if table:
        output.append("</tbody></table></div>")
    return "\n".join(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "_site")
    args = parser.parse_args()
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    import os
    for relative in PAGES:
        target = destination / target_path(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        links = []
        for page in PAGES:
            label = "Home" if page == Path("README.md") else page.stem.title()
            if page.parts[0] == "examples":
                label = {"cpu_mock": "CPU mock", "iteration_handoff": "FP handoff"}[page.parts[1]]
            url = os.path.relpath(destination / target_path(page), target.parent)
            links.append(f'<a href="{html.escape(url)}">{html.escape(label)}</a>')
        body = render((ROOT / relative).read_text(encoding="utf-8"))
        title = (ROOT / relative).read_text(encoding="utf-8").splitlines()[0].lstrip("# ")
        document = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        document += '<title>' + html.escape(title) + ' · DICE</title><style>'
        document += 'body{font:17px/1.65 system-ui,sans-serif;color:#172a3a;max-width:1050px;margin:0 auto;padding:30px}nav{display:flex;gap:15px;flex-wrap:wrap;border-bottom:1px solid #ccd4db;padding-bottom:18px}a{color:#145b89}h1{font-size:2.3em}h1,h2,h3{line-height:1.2}code,pre{font-family:ui-monospace,monospace;font-size:.9em}pre{background:#edf2f6;padding:18px;overflow:auto}code{background:#edf2f6}table{border-collapse:collapse;min-width:650px}td{border:1px solid #ccd4db;padding:10px;vertical-align:top}tr:first-child{font-weight:600}.table{overflow:auto}footer{border-top:1px solid #ccd4db;margin-top:40px;padding-top:12px;color:#526579}'
        document += '</style><nav aria-label="Documentation">' + ' '.join(links) + '</nav><main>' + body + '</main><footer>DICE · release documentation 2026-11-02</footer></html>'
        target.write_text(document, encoding="utf-8")
    # Keep the documented schema example available in the standalone HTML tree.
    import shutil
    data_asset = destination / "examples/data/labelled_h2.extxyz"
    data_asset.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / "examples/data/labelled_h2.extxyz", data_asset)
    (destination / ".nojekyll").touch()
    print(f"Built {len(PAGES)} static documentation pages in {destination}")


if __name__ == "__main__":
    main()
