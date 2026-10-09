#!/usr/bin/env python3
"""Split this workspace into a small GitHub snapshot and a Drive file list."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
import time


GITHUB_LIMIT = 2 * 1024 * 1024
DRIVE_EXTENSIONS = {
    ".h5", ".hdf5", ".bin", ".pt", ".pth", ".ckpt", ".npy", ".npz",
    ".safetensors", ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z",
}
METADATA = ["backup/manifest.json", "backup/github_files.txt", "backup/drive_files.txt"]


def atomic_write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".backup-tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_manifest(root, manifest):
    atomic_write(root / METADATA[0], json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


def relative_path(root, path):
    relative = path.relative_to(root).as_posix()
    if "\n" in relative or "\r" in relative:
        raise ValueError(f"Filename cannot be represented in an rclone file list: {relative!r}")
    return relative


def source_path(root, relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe manifest path: {relative!r}")
    return root / path


def walk_error(error):
    raise error


def classify(path, size):
    name = path.name.lower()
    if "__pycache__" in path.parts or path.suffix.lower() == ".pyc":
        return "skip", "regenerable Python bytecode"
    if (name == ".env" or name.startswith(".env.") and name not in
            {".env.example", ".env.sample", ".env.template"}) or name in {
            "credentials.json", "service-account.json", "service_account.json",
            "id_rsa", "id_ed25519"} or path.suffix.lower() in {".p12", ".key"}:
        return "skip", "potential private credential configuration"
    if path.suffix.lower() in DRIVE_EXTENSIONS:
        return "drive", "feature, model, array, checkpoint, or archive"
    if size > GITHUB_LIMIT:
        return "drive", "larger than 2 MiB"
    if not path.suffix:
        with path.open("rb") as stream:
            sample = stream.read(8192)
        try:
            sample.decode("utf-8")
        except UnicodeDecodeError:
            return "drive", "extensionless binary"
        if b"\0" in sample:
            return "drive", "extensionless binary"
    return "github", "small source, configuration, documentation, or result"


def prepare(root):
    files, symlinks, skipped, empty_directories = [], [], [], []
    for directory, directories, names in os.walk(root, followlinks=False, onerror=walk_error):
        base = Path(directory)
        if not directories and not names and base != root:
            empty_directories.append(relative_path(root, base))
        for name in list(directories):
            path = base / name
            if name in {".git", ".backup_work"}:
                skipped.append({"path": relative_path(root, path), "type": "directory",
                                "reason": "Git history or local backup working directory"})
                directories.remove(name)
            elif path.is_symlink():
                names.append(name)
                directories.remove(name)
        for name in names:
            path = base / name
            relative = relative_path(root, path)
            if relative in METADATA or name.startswith(".backup-tmp-"):
                continue
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                target = path.resolve(strict=False)
                if not target.is_relative_to(root):
                    raise ValueError(f"Symlink points outside the workspace: {relative}")
                symlinks.append({"path": relative, "original_target": os.readlink(path),
                                 "portable_target": os.path.relpath(target, path.parent)})
                continue
            if not stat.S_ISREG(info.st_mode):
                skipped.append({"path": relative, "type": "special",
                                "reason": "not a regular file or symbolic link"})
                continue
            destination, reason = classify(path, info.st_size)
            record = {"path": relative, "size": info.st_size, "mtime_ns": info.st_mtime_ns,
                      "mode": stat.S_IMODE(info.st_mode), "destination": destination, "reason": reason}
            files.append(record)
            if destination == "skip":
                skipped.append({"path": relative, "type": "file", "reason": reason})
    files.sort(key=lambda record: record["path"])
    symlinks.sort(key=lambda record: record["path"])
    summary = {destination: {
        "files": sum(record["destination"] == destination for record in files),
        "bytes": sum(record["size"] for record in files if record["destination"] == destination),
    } for destination in ("github", "drive", "skip")}
    summary["symlinks"] = len(symlinks)
    manifest = {"version": 1, "root": str(root), "created_at_utc":
                time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "github_max_bytes": GITHUB_LIMIT, "files": files, "symlinks": symlinks,
                "skipped": skipped, "empty_directories": sorted(empty_directories),
                "metadata_files": METADATA, "summary": summary}
    github = [record["path"] for record in files if record["destination"] == "github"]
    github += [record["path"] for record in symlinks] + METADATA
    drive = [record["path"] for record in files if record["destination"] == "drive"]
    atomic_write(root / METADATA[1], "".join(path + "\n" for path in sorted(github)))
    atomic_write(root / METADATA[2], "".join(path + "\n" for path in drive))
    save_manifest(root, manifest)
    print(json.dumps(summary, indent=2))
    return manifest


def load_manifest(root):
    with (root / METADATA[0]).open(encoding="utf-8") as stream:
        return json.load(stream)


def stable_stat(path, record):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or (info.st_size, info.st_mtime_ns, stat.S_IMODE(info.st_mode)) != (
            record["size"], record["mtime_ns"], record["mode"]):
        raise RuntimeError(f"Source changed since prepare; run prepare again: {record['path']}")
    return info


def stage(root, output):
    manifest = load_manifest(root)
    destination = output.resolve() / "won_woo"
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite an existing snapshot: {destination}")
    destination.mkdir(parents=True)
    selected = [record for record in manifest["files"] if record["destination"] == "github"]
    for record in selected:
        source = source_path(root, record["path"])
        before = stable_stat(source, record)
        target = source_path(destination, record["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        after = stable_stat(source, record)
        if (before.st_ino, before.st_dev) != (after.st_ino, after.st_dev) or target.stat().st_size != record["size"]:
            raise RuntimeError(f"Source changed while staging: {record['path']}")
    for record in manifest["symlinks"]:
        source = source_path(root, record["path"])
        if not source.is_symlink() or os.readlink(source) != record["original_target"]:
            raise RuntimeError(f"Symlink changed since prepare: {record['path']}")
        target = source_path(destination, record["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(record["portable_target"])
    for relative in manifest["empty_directories"]:
        source_path(destination, relative).mkdir(parents=True, exist_ok=True)
    for relative in manifest["metadata_files"]:
        target = source_path(destination, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path(root, relative), target)
    print(f"Staged {len(selected)} regular files and {len(manifest['symlinks'])} links in {destination}")
    return destination


def hash_files(root, target):
    manifest = load_manifest(root)
    selected = [record for record in manifest["files"] if record["destination"] != "skip"
                and (target == "all" or record["destination"] == target)]
    for index, record in enumerate(selected, 1):
        path = source_path(root, record["path"])
        before = stable_stat(path, record)
        if record.get("sha256"):
            continue
        digest = hashlib.sha256()
        total, last_report = 0, time.monotonic()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
                total += len(chunk)
                if time.monotonic() - last_report >= 10:
                    print(f"Hashing {record['path']}: {total}/{record['size']} bytes", flush=True)
                    last_report = time.monotonic()
        after = stable_stat(path, record)
        if (before.st_ino, before.st_dev) != (after.st_ino, after.st_dev) or total != record["size"]:
            raise RuntimeError(f"Source changed while hashing: {record['path']}")
        record["sha256"] = digest.hexdigest()
        save_manifest(root, manifest)
        print(f"[{index}/{len(selected)}] Hashed {record['path']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    staging = commands.add_parser("stage")
    staging.add_argument("--output", type=Path, default=Path(".backup_work/github_repository"))
    hashing = commands.add_parser("hash")
    hashing.add_argument("--target", choices=("github", "drive", "all"), default="drive")
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    if args.command == "prepare":
        prepare(root)
    elif args.command == "stage":
        stage(root, args.output)
    else:
        hash_files(root, args.target)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Backup error: {error}", file=sys.stderr)
        sys.exit(1)
