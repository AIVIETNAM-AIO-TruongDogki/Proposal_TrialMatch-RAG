"""Phase 0 — re-download rawdata/ on a fresh machine, from Hugging Face Hub.

    python -m src.fetch_data

rawdata/ (8.9GB: ClinicalTrials.gov 2021-04-27 snapshot + TREC 2021/2022
topics/qrels) is gitignored. Originally planned to re-fetch from trec-cds.org's
5 original zip files, but this project settled on a simpler path: **one
`rawdata.tar.gz` file**, tarred and uploaded by hand to a private Hugging Face
dataset repo, then downloaded and extracted by this script.

ONE-TIME PREP (on a machine that already has rawdata/) — see also
docs/decisions/data-fetch-recovery.md:
    tar -czf rawdata.tar.gz rawdata/      # run from the PROJECT ROOT, not inside rawdata/
    hf auth login
    hf upload <repo_name> rawdata.tar.gz --repo-type=dataset
    # set HF_DATASET_REPO=<repo_name> (and HF_TOKEN if private) in .env

Why not `hf upload . --repo-type=dataset` from the project root: `hf upload`
doesn't respect .gitignore automatically (only manual --include/--exclude) —
running it from root would try to upload .env (leaking the API key), .venv/
(5.5GB), .git/, data/, indexes/. Always tar to ONE file and upload that.

The archive already has a `rawdata/` prefix inside it (that's what `tar -czf
rawdata.tar.gz rawdata/` produces), so it's extracted into the PARENT of
--dest, not into --dest itself.
"""

from __future__ import annotations

import argparse
import os
import sys
import tarfile
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

load_dotenv()

ARCHIVE_NAME = "rawdata.tar.gz"
EXPECTED_NCT_COUNT = 375_580  # see CLAUDE.md — find rawdata -name 'NCT*.xml' | wc -l


class FetchError(RuntimeError):
    pass


def _looks_populated(dest: Path) -> bool:
    return (dest / "topics2021.xml").exists()


def extract_archive(archive_path: str, dest: Path) -> None:
    """Extract a .tar.gz that already has a rawdata/ prefix into dest's parent.

    Kept separate from fetch() so it's testable with a fake archive, without
    needing real Hugging Face access.
    """
    with tarfile.open(archive_path) as tf:
        tf.extractall(dest.parent)


def verify_count(dest: Path) -> int:
    """Count NCT*.xml files — measured, not assumed. See CLAUDE.md."""
    count = sum(1 for _ in dest.rglob("NCT*.xml"))
    if count != EXPECTED_NCT_COUNT:
        print(f"CANH BAO: tim thay {count} file NCT*.xml, ky vong {EXPECTED_NCT_COUNT}. "
              f"Kiem tra lai noi dung archive.", file=sys.stderr)
    else:
        print(f"OK: {count} file NCT*.xml, khop voi ky vong.")
    return count


def fetch(dest: str = "rawdata", repo_id: str | None = None, token: str | None = None,
          force: bool = False) -> None:
    dest_path = Path(dest)
    if not force and _looks_populated(dest_path):
        print(f"{dest_path} co ve da co du lieu (thay {dest_path / 'topics2021.xml'}). "
              f"Bo qua tai — dung --force de tai lai.")
        return

    repo_id = repo_id or os.environ.get("HF_DATASET_REPO")
    if not repo_id:
        raise FetchError(
            "Chua cau hinh HF_DATASET_REPO (hoac --hf-repo). Chuan bi truoc khi chay lai:\n"
            "  1. tar -czf rawdata.tar.gz rawdata/   (tren may DA CO san rawdata/, chay tu goc du an)\n"
            "  2. hf auth login\n"
            "  3. hf upload <ten_repo> rawdata.tar.gz --repo-type=dataset\n"
            "  4. Dat HF_DATASET_REPO=<ten_repo> (va HF_TOKEN neu repo private) trong .env\n"
            "Chi tiet: docs/decisions/data-fetch-recovery.md")

    token = token or os.environ.get("HF_TOKEN") or None

    print(f"Tai {ARCHIVE_NAME} tu dataset repo '{repo_id}' tren Hugging Face Hub...")
    try:
        archive_path = hf_hub_download(
            repo_id=repo_id, filename=ARCHIVE_NAME, repo_type="dataset", token=token)
    except HfHubHTTPError as e:
        raise FetchError(f"Khong tai duoc {ARCHIVE_NAME} tu '{repo_id}': {e}") from e

    print(f"Giai nen vao {dest_path.parent or '.'}/ ...")
    extract_archive(archive_path, dest_path)
    verify_count(dest_path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="rawdata")
    ap.add_argument("--hf-repo", default=None, help="mac dinh doc HF_DATASET_REPO tu .env")
    ap.add_argument("--hf-token", default=None, help="mac dinh doc HF_TOKEN tu .env (chi can neu repo private)")
    ap.add_argument("--force", action="store_true", help="tai lai du dest da co du lieu")
    args = ap.parse_args()

    try:
        fetch(args.dest, args.hf_repo, args.hf_token, args.force)
    except FetchError as e:
        print(str(e), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
