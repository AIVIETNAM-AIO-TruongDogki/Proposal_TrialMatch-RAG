"""Phase 0 — tai lai rawdata/ tren mot may moi, tu Hugging Face Hub.

    python -m src.fetch_data

rawdata/ (8.9 GB: snapshot ClinicalTrials.gov 2021-04-27 + topics/qrels TREC 2021/2022) khong nam
trong git (xem .gitignore). Ban dau du dinh tai lai tu 5 file zip goc cua trec-cds.org, nhung du an
nay dung lai o mot phuong an don gian hon: **mot file `rawdata.tar.gz` duy nhat**, do nguoi dung tu
tar + upload len mot Hugging Face dataset repo rieng, roi script nay tai va giai nen lai.

CHUAN BI (mot lan, tren may da co san rawdata/) — xem them docs/decisions/data-fetch-recovery.md:
    tar -czf rawdata.tar.gz rawdata/      # chay tu GOC du an, KHONG phai tu trong rawdata/
    hf auth login
    hf upload <ten_repo> rawdata.tar.gz --repo-type=dataset
    # dat HF_DATASET_REPO=<ten_repo> (va HF_TOKEN neu repo private) trong .env

Vi sao khong `hf upload . --repo-type=dataset` tu goc du an: `hf upload` khong tu dong loai tru theo
.gitignore (chi co --include/--exclude thu cong) — chay tu goc se co gang day ca .env (lo API key),
.venv/ (5.5 GB), .git/, data/, indexes/. Luon nen thanh MOT file va upload dung file do.

Archive mang san tien to `rawdata/` ben trong (dung `tar -czf rawdata.tar.gz rawdata/` la ra dung
the), nen giai nen vao THU MUC CHA cua --dest, khong phai vao chinh --dest.
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
EXPECTED_NCT_COUNT = 375_580  # xem CLAUDE.md — find rawdata -name 'NCT*.xml' | wc -l


class FetchError(RuntimeError):
    pass


def _looks_populated(dest: Path) -> bool:
    return (dest / "topics2021.xml").exists()


def extract_archive(archive_path: str, dest: Path) -> None:
    """Giai nen mot file .tar.gz mang san tien to rawdata/ vao thu muc cha cua dest.

    Tach rieng ham nay khoi fetch() de kiem thu duoc bang mot archive gia lap, khong can
    Hugging Face that.
    """
    with tarfile.open(archive_path) as tf:
        tf.extractall(dest.parent)


def verify_count(dest: Path) -> int:
    """Dem file NCT*.xml — do luong, khong gia dinh thanh cong. Xem CLAUDE.md."""
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
