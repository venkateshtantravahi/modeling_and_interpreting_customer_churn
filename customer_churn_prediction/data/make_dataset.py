from __future__ import annotations

import argparse
import json
import os
import stat
import time
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from kaggle.api.kaggle_api_extended import KaggleApi

LOGGER = logging.getLogger(__name__)
DEFAULT_OUT_DIR = Path("data/raw")


def _sha256(path: Path) -> str:
    """Compute a file's SHA-256 checksum in streaming mode."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json_atomic(path: Path, payload: dict) -> None:
    """Write JSON atomically to avoid partial files in concurrent scenarios."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


@dataclass(frozen=True)
class KaggleCredentials:
    """Holds Kaggle credentials"""
    username: Optional[str] = None
    key: Optional[str] = None

    @classmethod
    def from_env(cls) -> "KaggleCredentials":
        return cls(
            username=os.getenv("KAGGLE_USERNAME"),
            key=os.getenv("KAGGLE_API_KEY"),
        )


class KaggleDatasetIngestor:
    """
    Download and unzip kaggle dataset into a target directory with a manifest.

    Features:
    - Idempotent by default (skips if manifest exists and files already present)
    - Optional `force=True` to re-download
    - Uses ~/.kaggle/kaggle.json by default, or ENV variables.
    - Writes data/raw/manifest.json
    """

    def __init__(
            self,
            dataset_slug: str,
            out_dir: Path = DEFAULT_OUT_DIR,
            unzip: bool = True,
            quiet: bool = False,
            creds: Optional[KaggleCredentials] = None,
    ) -> None:
        """
        Args:
            dataset_slug: e.g. "imsparsh/churn-risk-rate-hackerearth-ml"
            out_dir: destination directory (default: data/raw)
            unzip: unzip the dataset after download (recommended True)
            quiet: pass 'quiet' to Kaggle API to reduce verbosity
            creds: optional credentials; if provided, will be used to create a temp kaggle config
        """
        self.dataset_slug = dataset_slug
        self.out_dir = out_dir
        self.unzip = unzip
        self.quiet = quiet
        self.creds = creds or KaggleCredentials.from_env()

        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.out_dir / "manifest.json"

    # ------------------- Public API ------------------------

    def download(self, force: bool = False) -> Dict:
        """
        Download dataset if needed, compute manifest, and return it.

        Returns:
            manifest dict with dataset slug, timestamp, and file metadata.
        """
        if self._is_already_downloaded() and not force:
            LOGGER.info(
                "Dataset already present, skipping download (use force=True to re-download).")
            return self._read_manifest()

        LOGGER.info("Starting Kaggle dataset download: %s",
                    self.dataset_slug)
        self._authenticate_and_download()
        manifest = self._build_manifest()
        _write_json_atomic(self.manifest_path, manifest)
        LOGGER.info(
            "Download complete: %d files -> %s",
            len(manifest["files"]),
            self.out_dir.as_posix(),
        )
        return manifest

    def _is_already_downloaded(self) -> bool:
        if not self.manifest_path.exists():
            return False
        try:
            manifest = self._read_manifest()
        except Exception:
            return False
        # consider it "present" if at least one listed file exists
        return any(Path(f["path"]).exists() for f in manifest.get("files", []))

    def _read_manifest(self) -> Dict:
        return json.loads(self.manifest_path.read_text())

    def _authenticate_and_download(self) -> None:
        """
        Auth flow:
          - If KAGGLE_USERNAME & KAGGLE_KEY envs are set, create a runtime config dir and
            write kaggle.json there (0600 perms). Kaggle API will pick it up via KAGGLE_CONFIG_DIR.
          - Else rely on ~/.kaggle/kaggle.json.
        """
        cleanup_dir: Optional[Path] = None
        try:
            if self.creds.username and self.creds.key:
                cfg_dir = Path(".kaggle_runtime")
                cfg_dir.mkdir(parents=True, exist_ok=True)
                kaggle_json = cfg_dir / "kaggle.json"
                kaggle_json.write_text(
                    json.dumps({"username": self.creds.username,
                                "key": self.creds.key})
                )
                kaggle_json.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0o600
                os.environ["KAGGLE_CONFIG_DIR"] = str(cfg_dir.resolve())
                cleanup_dir = cfg_dir  # remember to clean up

            api = KaggleApi()
            api.authenticate()

            # NOTE: dataset_download_files downloads a zip and unzips (if unzip=True)
            # into the provided path. It will overwrite files if present.
            api.dataset_download_files(
                self.dataset_slug,
                path=str(self.out_dir),
                unzip=self.unzip,
                quiet=self.quiet,
            )
        finally:
            # Clean up temporary runtime config if we created one
            if cleanup_dir and cleanup_dir.exists():
                try:
                    for p in cleanup_dir.glob("*"):
                        p.unlink(missing_ok=True)
                    cleanup_dir.rmdir()
                except Exception:
                    # non-fatal if cleanup fails in CI
                    pass

    def _build_manifest(self) -> Dict:
        files: List[Path] = [
            p
            for p in self.out_dir.rglob("*")
            if p.is_file() and p.name != self.manifest_path.name
        ]
        file_meta = [
            {"path": str(p), "size": p.stat().st_size, "sha256": _sha256(p)}
            for p in files
        ]
        return {
            "dataset": self.dataset_slug,
            "downloaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "unzip": self.unzip,
            "files": file_meta,
        }


def _setup_logging(verbosity: int) -> None:
    level = logging.WARNING if verbosity == 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download/unzip a Kaggle dataset into data/raw with a manifest."
    )
    parser.add_argument(
        "--dataset",
        default="imsparsh/churn-risk-rate-hackerearth-ml",
        help='Kaggle dataset slug (e.g. "imsparsh/churn-risk-rate-hackerearth-ml")',
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT_DIR),
        help="Output directory (default: data/raw)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if a manifest already exists",
    )
    parser.add_argument(
        "--no-unzip",
        dest="unzip",
        action="store_false",
        help="Do not unzip (keep downloaded archive only)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Reduce Kaggle API verbosity",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=1,
        help="Increase log verbosity (-v INFO, -vv DEBUG; default INFO)",
    )
    return parser.parse_args(argv)

def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    _setup_logging(args.verbose)

    out_dir = Path(args.out)
    ingestor = KaggleDatasetIngestor(
        dataset_slug=args.dataset,
        out_dir=out_dir,
        unzip=args.unzip,
        quiet=args.quiet,
    )
    manifest = ingestor.download(force=args.force)
    LOGGER.info("Wrote manifest: %s", (out_dir / "manifest.json").as_posix())
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()