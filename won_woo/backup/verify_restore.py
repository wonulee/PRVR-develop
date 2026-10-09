#!/usr/bin/env python3
"""Verify a restored workspace against the backup SHA256 manifest."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import time


def safe_path(root, relative):
    if not isinstance(relative, str) or not relative:
        raise ValueError("Manifest path must be a nonempty string")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe relative path: {relative!r}")
    target = root / path
    if not target.resolve(strict=False).is_relative_to(root):
        raise ValueError(f"Path resolves outside the workspace: {relative!r}")
    return target


def verify(root, manifest_path):
    root = root.resolve(strict=True)
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), list):
        raise ValueError("Manifest must contain a files list")
    errors = []
    counts = {"github": {"files": 0, "bytes": 0}, "drive": {"files": 0, "bytes": 0}}
    seen = set()
    for record in manifest["files"]:
        if not isinstance(record, dict):
            errors.append("File record must be an object")
            continue
        destination = record.get("destination")
        if destination == "skip":
            continue
        relative = record.get("path")
        label = str(relative)
        try:
            if destination not in counts:
                raise ValueError(f"Unknown destination: {destination!r}")
            path = safe_path(root, relative)
            if relative in seen:
                raise ValueError("Duplicate manifest path")
            seen.add(relative)
            expected = record.get("sha256")
            if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
                raise ValueError("Missing or invalid SHA256 digest")
            size = record.get("size")
            if type(size) is not int or size < 0:
                raise ValueError("Missing or invalid expected file size")
            if not path.exists():
                raise ValueError("Missing file")
            before = path.lstat()
            if not stat.S_ISREG(before.st_mode):
                raise ValueError("Expected a regular file")
            if before.st_size != size:
                raise ValueError(f"Size mismatch: expected {size}, found {before.st_size}")
            digest, total = hashlib.sha256(), 0
            last_report = time.monotonic()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
                    total += len(chunk)
                    if time.monotonic() - last_report >= 10:
                        print(f"Verifying {relative}: {total}/{size} bytes", flush=True)
                        last_report = time.monotonic()
            after = path.lstat()
            if (before.st_size, before.st_mtime_ns, before.st_ino, before.st_dev) != (
                    after.st_size, after.st_mtime_ns, after.st_ino, after.st_dev) or total != size:
                raise ValueError("File changed during verification")
            if digest.hexdigest() != expected.lower():
                raise ValueError("SHA256 mismatch")
            counts[destination]["files"] += 1
            counts[destination]["bytes"] += size
        except (OSError, ValueError) as error:
            errors.append(f"{label}: {error}")
    links = manifest.get("symlinks", [])
    if not isinstance(links, list):
        raise ValueError("Manifest symlinks must be a list")
    verified_links = 0
    for record in links:
        if not isinstance(record, dict):
            errors.append("Symlink record must be an object")
            continue
        relative = record.get("path")
        try:
            path = safe_path(root, relative)
            if relative in seen:
                raise ValueError("Duplicate manifest path")
            seen.add(relative)
            expected = record.get("portable_target")
            if not isinstance(expected, str) or not expected or Path(expected).is_absolute():
                raise ValueError("Missing or invalid portable symlink target")
            if not (path.parent / expected).resolve(strict=False).is_relative_to(root):
                raise ValueError("Symlink target resolves outside the workspace")
            if not path.is_symlink():
                raise ValueError("Missing symlink")
            actual = os.readlink(path)
            if actual != expected:
                raise ValueError(f"Symlink target mismatch: expected {expected!r}, found {actual!r}")
            if not path.exists():
                raise ValueError(f"Broken symlink target: {expected!r}")
            verified_links += 1
        except (OSError, ValueError) as error:
            errors.append(f"{relative}: {error}")
    if errors:
        raise RuntimeError("Restore verification failed:\n" + "\n".join(errors))
    counts["symlinks"] = verified_links
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    counts = verify(root, args.manifest or root / "backup/manifest.json")
    files = counts["github"]["files"] + counts["drive"]["files"]
    size = counts["github"]["bytes"] + counts["drive"]["bytes"]
    print(f"Restore verified: {files} files, {size} bytes, {counts['symlinks']} symlinks")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
