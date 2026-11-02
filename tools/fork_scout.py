"""Inventory public GitHub forks and identify possession-method evidence in changed source.

Python 3.8+, Git. No fork code is checked out or executed. Run with --help.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT = "metrica-sports/more-than-a-hack-2027"
TEAM = "Tejeshhh78/MesQueUnCode"
# Discovery cues, not quality scores. Review the linked source before drawing conclusions.
CUES = {
    "Zustandsdekodierung": r"viterbi|hidden.markov|semi.markov",
    "Balllücken": r"interpolat|kalman|fill_short_gaps",
    "Ballkontrolle": r"control_radius|nearest.player|time.to.intercept|hysteres|duel_margin",
    "Spielunterbrechung": r"dead.ball|out.of.play|restart|set.piece",
    "Tagging": r"smart.tag|tagging_evidence",
    "Kameras": r"broadcast|multi.view|dual.view",
    "Validierung": r"ablation|holdout|held.out|cross.valid",
}
RULES = [(name, re.compile(pattern, re.I)) for name, pattern in CUES.items()]
SUFFIXES = {".py", ".ipynb", ".r", ".jl", ".cpp", ".ts", ".js"}
LIMIT = 1024 * 1024


def github(url, limit=None):
    headers = {"User-Agent": "MesQueUnCode-fork-scout"}
    if url.startswith("https://api.github.com/") and os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = "Bearer " + os.environ["GITHUB_TOKEN"]
    for attempt in range(3):
        try:
            with urlopen(Request(url, headers=headers), timeout=35) as res:
                return res.read(limit), res.headers
        except HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == 2:
                raise
        except OSError:
            if attempt == 2:
                raise
        time.sleep(2 ** attempt)


def forks(root):
    """GitHub forks_count includes descendants. Visit non-leaf parents, follow pagination."""
    queue, seen, result, errors = [root], {root.casefold()}, [], []
    while queue:
        parent = queue.pop()
        url = f"https://api.github.com/repos/{parent}/forks?per_page=100&sort=oldest"
        try:
            while url:
                data, headers = github(url)
                for item in json.loads(data):
                    name = item["full_name"]
                    if name.casefold() in seen:
                        continue
                    seen.add(name.casefold())
                    result.append(name)
                    if item["forks_count"]:
                        queue.append(name)
                link = re.search(r'<(https://api\.github\.com/[^>]+)>; rel="next"', headers.get("Link", ""))
                url = link.group(1) if link else None
        except (OSError, ValueError) as exc:
            errors.append(f"{parent}: {exc}")
    return sorted(result), errors


def git(*args, cwd=None):
    p = subprocess.run(["git", *args], cwd=cwd, capture_output=True, timeout=150,
                       env={**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_OPTIONAL_LOCKS": "0"})
    if p.returncode:
        raise ValueError(p.stderr.decode(errors="replace").strip())
    return p.stdout


def tree(folder, rev):
    records = git("ls-tree", "-rz", rev, cwd=folder).decode("utf-8", "replace").split("\0")
    result = {}
    for record in filter(None, records):
        meta, path = record.split("\t", 1)
        mode, _, sha = meta.split()
        if mode in {"100644", "100755"}:
            result[path] = sha
    return result


def source_lines(data, path):
    src = data.decode("utf-8-sig", "replace")
    if path.lower().endswith(".ipynb"):
        notebook = json.loads(src)
        if not isinstance(notebook, dict):
            raise ValueError("Invalid notebook")
        src = "\n".join("".join(cell.get("source", [])) for cell in notebook.get("cells", [])
                        if isinstance(cell, dict) and cell.get("cell_type") == "code")
    return src.splitlines()


def evidence(name, head, path, inherited):
    urlpath = quote(path, safe="/")
    data, _ = github(f"https://raw.githubusercontent.com/{name}/{head}/{urlpath}", LIMIT + 1)
    if len(data) > LIMIT:
        return {"skip": "over 1 MiB"}
    found = {}
    for number, line in enumerate(source_lines(data, path), 1):
        if line.strip() in inherited:
            continue
        for label, pattern in RULES:
            if label not in found and pattern.search(line):
                found[label] = number
    link = f"https://github.com/{name}/blob/{head}/{urlpath}"
    return {"hits": {label: link if path.lower().endswith(".ipynb") else f"{link}#L{line}"
                     for label, line in found.items()}}


def scan(name, known, inherited_blobs, inherited_lines, scratch):
    row = {"repo": name, "own": name.lower() == TEAM.lower(), "branches": {}, "changes": {}, "errors": []}
    try:
        url = f"https://github.com/{name}.git"
        refs = git("ls-remote", "--heads", url).decode().splitlines()
        for line in refs:
            sha, ref = line.split("\t")
            row["branches"][ref[len("refs/heads/"):]] = sha
        heads = sorted(set(row["branches"].values()) - known)
        if not heads:
            return row
        with tempfile.TemporaryDirectory(dir=scratch) as temp:
            # Only trees and commit history. Raw source is read separately with a size cap.
            git("init", "--bare", "--quiet", temp)
            git("fetch", "--quiet", "--filter=blob:none", "--no-tags", url, *heads, cwd=temp)
            blobs = {}
            for head in heads:
                files = tree(temp, head)
                paths = {p: sha for p, sha in files.items() if sha not in inherited_blobs}
                details = {"nonstarter_files": sorted(paths), "sources": {}}
                row["changes"][head] = details
                for path, sha in paths.items():
                    parts = {p.lower() for p in Path(path).parts[:-1]}
                    if parts & {"data", "node_modules", ".venv", "vendor", "gt", "site-packages"}:
                        continue
                    if Path(path).suffix.lower() not in SUFFIXES:
                        continue
                    key = (sha, path)
                    if key not in blobs:
                        try:
                            blobs[key] = evidence(name, head, path, inherited_lines)
                        except (OSError, ValueError, TypeError) as exc:
                            row["errors"].append(f"{head[:8]} {path}: {exc}")
                            continue
                    details["sources"][path] = blobs[key]
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        row["errors"].append(str(exc))
    return row


def markdown(report):
    rows = report["forks"]
    active = [row for row in rows if row["changes"]]
    errors = report["discovery_errors"] + [f"{row['repo']}: {msg}" for row in rows for msg in row["errors"]]
    lines = ["# Öffentliche Challenge-Forks", "", f"Stand {report['utc']}.",
             f"{len(rows)} Forks, {sum(len(row['branches']) for row in rows)} Branches. "
             f"{len(active)} Forks haben mindestens eine untersuchte Branch-Spitze außerhalb der Starter-Historie.",
             "", "Die Tabelle zählt Forks einmal pro Thema. Ein Treffer ist ein Lesehinweis in verändertem "
             "Quelltext. Auch Kommentare, Tests und aufgegebene Experimente können Treffer erzeugen. "
             "Private Scores sind unbekannt. Euer Repo ist aus den Themenzahlen ausgenommen.", "",
             "| Thema | Forks | Beispielstelle |", "|---|---:|---|"]
    all_hits = {}
    for row in active:
        if row["own"]:
            continue
        for change in row["changes"].values():
            for source in change["sources"].values():
                for label, link in source.get("hits", {}).items():
                    all_hits.setdefault(label, {}).setdefault(row["repo"], link)
    for label, matches in sorted(all_hits.items(), key=lambda item: (-len(item[1]), item[0])):
        name, link = next(iter(matches.items()))
        lines.append(f"| {label} | {len(matches)} | [{name}]({link}) |")
    lines += ["", "## Abdeckung", "", "GitHub-Forknetz einschließlich Unterforks. Alle sichtbaren "
              "Branch-Spitzen werden mit allen historischen Starter-Dateiinhalten verglichen. "
              "Der Scanner durchsucht ausgewählte Codeformate bis 1 MiB. Die JSON-Datei enthält "
              "sämtliche Forks, Branches, zusätzliche Dateipfade, ausgelassene große Quellen und Fundstellen.",
              "", f"Fehler beim Abruf: {len(errors)}."]
    lines += [f"- {error}" for error in errors]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/fork-scout.json"))
    parser.add_argument("--jobs", type=int, choices=range(1, 17), default=6)
    args = parser.parse_args()
    names, errors = forks(ROOT)
    print(f"{len(names)} public forks discovered", flush=True)
    with tempfile.TemporaryDirectory() as scratch:
        base = Path(scratch) / "upstream"
        git("clone", "--quiet", "--bare", f"https://github.com/{ROOT}.git", str(base))
        known = set(git("rev-list", "--branches", cwd=base).decode().splitlines())
        inherited_blobs, inherited_lines = set(), set()
        for sha in known:
            for path, blob in tree(base, sha).items():
                if blob in inherited_blobs:
                    continue
                inherited_blobs.add(blob)
                if Path(path).suffix.lower() in SUFFIXES:
                    data = git("cat-file", "blob", blob, cwd=base)
                    inherited_lines.update(line.strip() for line in source_lines(data, path))
        def one(name):
            return scan(name, known, inherited_blobs, inherited_lines, scratch)
        with ThreadPoolExecutor(args.jobs) as executor:
            rows = []
            for i, row in enumerate(executor.map(one, names), 1):
                rows.append(row)
                if i % 20 == 0:
                    print(f"{i}/{len(names)}", flush=True)
        report = {"upstream": ROOT, "upstream_head": git("rev-parse", "HEAD", cwd=base).decode().strip(),
                  "utc": datetime.now(timezone.utc).isoformat(), "discovery_errors": errors, "forks": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.output.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    return int(bool(errors or any(row["errors"] for row in rows)))


if __name__ == "__main__":
    raise SystemExit(main())
