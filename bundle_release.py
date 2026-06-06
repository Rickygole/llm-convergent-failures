"""
bundle_release.py

Zip the icdm_release folder into a single .zip suitable for upload to a public
GitHub repository or for direct attachment to a paper submission portal.

Usage in Google Colab:

    from google.colab import drive
    drive.mount('/content/drive')
    !python "/content/drive/MyDrive/one army/icdm_release/bundle_release.py"

The script will produce `/content/icdm_release.zip` ready to download.

Usage locally:

    python bundle_release.py /path/to/icdm_release /path/to/output.zip
"""

from __future__ import annotations
import sys
import zipfile
from pathlib import Path


def bundle(release_root: Path, out_zip: Path) -> None:
    if not release_root.exists():
        raise FileNotFoundError(f"Release root not found: {release_root}")

    files_added = 0
    skipped = 0

    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in sorted(release_root.rglob("*")):
            if not p.is_file():
                continue
            # Skip the bundle script itself and any hidden files
            if p.name == "bundle_release.py" or p.name.startswith("."):
                skipped += 1
                continue
            arcname = p.relative_to(release_root.parent)  # keep the top-level dir name
            zf.write(p, arcname)
            files_added += 1

    print(f"Bundled {files_added} files into {out_zip}")
    print(f"Archive size: {out_zip.stat().st_size / 1024:.1f} KB")
    print(f"Skipped: {skipped} (bundle script and hidden files)")


def main() -> None:
    if len(sys.argv) >= 3:
        release_root = Path(sys.argv[1])
        out_zip = Path(sys.argv[2])
    else:
        # Defaults for the typical Colab + Drive layout
        release_root = Path("/content/drive/MyDrive/one army/icdm_release")
        out_zip = Path("/content/icdm_release.zip")

    bundle(release_root, out_zip)


if __name__ == "__main__":
    main()
