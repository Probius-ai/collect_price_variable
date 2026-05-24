"""Quarantine flow for files whose filename does not match their content.

Round 2: HOME_전력거래_계통한계가격_가중평균SMP.csv and the ` (2).csv` sibling
*claim* to be SMP but actually contain yearly generation breakdowns by
fuel source. We must NOT load them via the SMP loader; instead we move
them aside and record the mismatch in a quarantine manifest.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import shutil
import unicodedata
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

from src.config.settings import get_settings, get_source_config
from src.utils.io import manual_drop_dir
from src.utils.logging import get_logger
from src.utils.time import collected_now_utc

logger = get_logger(__name__)


@dataclass
class QuarantineEntry:
    source_id: str
    candidate_source_id: str
    status: str
    reason: str
    original_path: str
    quarantined_path: str
    sha256: str
    notes: list[str]

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def quarantine_dir() -> Path:
    """`data/raw/manual_or_filedata/quarantine/<candidate_source>/`."""
    return get_settings().data_dir / "raw" / "manual_or_filedata" / "quarantine"


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def quarantine_filename_content_mismatch(
    candidate_source_id: str = "kpx_generation_yearly",
    *,
    file_paths: Iterable[Path] | None = None,
) -> list[QuarantineEntry]:
    """Move misnamed files into the quarantine area and write a manifest.

    If `file_paths` is None, scans `data/raw/manual_or_filedata/` for files
    matching any pattern in the source's `filename_globs_misnamed` field.
    """
    config = get_source_config(candidate_source_id)
    if config.get("status") != "pending_schema_verification":
        raise ValueError(
            f"Source {candidate_source_id!r} is not flagged for quarantine "
            f"(status={config.get('status')!r})."
        )
    qdir = quarantine_dir() / candidate_source_id
    qdir.mkdir(parents=True, exist_ok=True)

    if file_paths is None:
        patterns = config.get("filename_globs_misnamed") or []
        drop_root = get_settings().data_dir / "raw" / "manual_or_filedata"
        # pathlib.Path.rglob is broken on Python 3.14 for patterns containing
        # non-ASCII characters (the bytes match but glob returns 0). We iterate
        # manually and apply NFC-normalised fnmatch on the basename.
        nfc_patterns = [unicodedata.normalize("NFC", p) for p in patterns]
        file_paths = []
        for root in [drop_root] if drop_root.exists() else []:
            for hit in sorted(root.rglob("*")):
                if not hit.is_file():
                    continue
                if "quarantine" in hit.parts:
                    continue
                name = unicodedata.normalize("NFC", hit.name)
                if any(fnmatch.fnmatch(name, pat) for pat in nfc_patterns):
                    file_paths.append(hit)

    stamp = collected_now_utc().strftime("%Y%m%dT%H%M%SZ")
    entries: list[QuarantineEntry] = []
    for src_path in file_paths:
        target = qdir / f"{stamp}__{src_path.name}"
        shutil.move(str(src_path), str(target))
        entry = QuarantineEntry(
            source_id=config.get("source_id", candidate_source_id),
            candidate_source_id=candidate_source_id,
            status=config["status"],
            reason=config.get("reason", "filename_content_mismatch"),
            original_path=str(src_path),
            quarantined_path=str(target),
            sha256=_sha256(target),
            notes=list(config.get("limitations") or []),
        )
        entries.append(entry)
        logger.warning(
            "Quarantined %s -> %s (reason=%s, source_id=%s)",
            src_path.name, target, entry.reason, candidate_source_id,
        )

    if entries:
        manifest = qdir / f"manifest_{stamp}.json"
        manifest.write_text(
            json.dumps([e.to_dict() for e in entries], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Wrote quarantine manifest: %s", manifest)
    return entries
