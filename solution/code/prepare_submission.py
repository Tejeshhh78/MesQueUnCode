"""Regenerate, validate and package the frozen submission, without scoring any labels.

    python -m solution.code.prepare_submission
"""

import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile

from evaluation import possession_metrics as pm
from .possession import release
from .possession.__main__ import ROOT, run, sha256

TECHNICAL_DOCUMENT = ROOT / "docs/Mes_Que_Un_Code.md"


def verify_output(output, manifest):
    """Verify the exact selected file, independently of the inference writer."""
    target = Path(output) / f"{release.TEAM}.csv"
    reason = pm.validate_submission(pm.read_label_csv(target), release.ROWS)
    if reason:
        raise ValueError(f"Submission validation failed: {reason}")
    if sha256(target) != release.SHA256 or manifest.get("submission_sha256") != release.SHA256:
        raise ValueError("Generated CSV differs from the frozen submission checksum")
    if manifest.get("profile") != release.PROFILE or manifest.get("status") != "selected":
        raise ValueError("Manifest does not describe the selected submission profile")
    if manifest.get("rows") != release.ROWS or manifest.get("ground_truth_used") is not False:
        raise ValueError("Invalid frame coverage or inference provenance")


def prepare(output_dir, data_dir=None):
    output = Path(output_dir)
    document = TECHNICAL_DOCUMENT.read_text(encoding="utf-8")
    if any(value not in document for value in (release.SHA256, release.PROFILE, release.BRANCH)):
        raise ValueError("Technical document does not match the frozen submission selection")
    with contextlib.redirect_stdout(io.StringIO()):
        manifest = run(data_dir, output, release.TEAM, profile=release.PROFILE)
    verify_output(output, manifest)
    with tempfile.TemporaryDirectory(prefix="mesqueuncode-reproduction-") as temporary:
        with contextlib.redirect_stdout(io.StringIO()):
            reproduced = run(data_dir, temporary, release.TEAM, profile=release.PROFILE)
        verify_output(temporary, reproduced)
    if any(manifest[key] != reproduced[key] for key in ("inputs_sha256", "code_sha256", "config")):
        raise ValueError("Inputs, source or configuration changed during reproduction")
    document_path = output / f"{release.TEAM}.md"
    if document_path.read_text(encoding="utf-8") != document:
        raise ValueError("Packaged technical document differs from the reviewed source")
    check = {
        "profile": release.PROFILE, "team": release.TEAM, "rows": release.ROWS,
        "submission_sha256": release.SHA256, "official_validation_passed": True,
        "reproduction_passed": True, "ground_truth_scored": False,
        "repository": release.REPOSITORY, "source_branch": release.BRANCH,
        "environment": manifest["environment"],
        "source_document_sha256": sha256(TECHNICAL_DOCUMENT),
        "public_source_publication_verified": False,
    }
    check_path = output / "preparation.json"
    check_path.write_text(json.dumps(check, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    files = [output / f"{release.TEAM}{suffix}" for suffix in (".csv", ".md", ".manifest.json")]
    files.append(check_path)
    (output / "SHA256SUMS.txt").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in files), encoding="utf-8")
    (output / "SUBMISSION.md").write_text(
        f"# Submission package: {release.TEAM}\n\n"
        f"Selected profile: `{release.PROFILE}`. CSV and reproduction verified locally.\n\n"
        f"Upload `{release.TEAM}.csv` and `{release.TEAM}.md` to the official form.\n"
        f"Repository: {release.REPOSITORY}, branch `{release.BRANCH}`.\n\n"
        "Before submitting, commit and publish the exact tested source branch and ensure the reviewer can access it. "
        "Publication and form submission are not performed by this command.\n\n"
        "The manifest, preparation report and checksums are local audit files. "
        "Keep tracking, caches and generated match data out of the public repository.\n",
        encoding="utf-8")
    print(f"Ready locally: {release.PROFILE}, {release.ROWS:,} rows, official validation and reproduction passed.")
    print(f"Submission files: {output / (release.TEAM + '.csv')} and {document_path}")
    print(f"SHA-256: {release.SHA256}")
    return check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "solution/results/final")
    parser.add_argument("--data-dir", help="Match data root; defaults to data/ or MTAH_DATA_DIR")
    args = parser.parse_args()
    try:
        prepare(args.output_dir, args.data_dir)
    except (ValueError, OSError) as error:
        parser.exit(1, f"ERROR: {error}\n")


if __name__ == "__main__":
    main()
