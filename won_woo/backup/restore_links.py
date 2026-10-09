#!/usr/bin/env python3
"""Repair links in a restored backup, using HDF5 metadata without reading arrays."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys


def workspace_path(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe manifest path: {relative}")
    path = root / relative
    if not path.parent.resolve().is_relative_to(root):
        raise ValueError(f"Manifest path leaves restored workspace: {relative}")
    return path


def restored_target(filename, parent, old_root, root):
    target = Path(filename)
    if target.is_absolute():
        if not target.is_relative_to(old_root):
            raise ValueError(f"Absolute target outside original workspace: {filename}")
        target = root / target.relative_to(old_root)
    else:
        target = parent / target
    if not target.resolve().is_relative_to(root):
        raise ValueError(f"Target leaves restored workspace: {filename}")
    if not target.exists():
        raise ValueError(f"Missing restored target: {target}")
    return target


def repair(root, manifest, allow_source_root=False):
    root = Path(root).resolve(strict=True)
    old_root = Path(manifest.get("source_root", manifest.get("root", "")))
    if not old_root.is_absolute():
        raise ValueError("Manifest requires an absolute root field")
    if root == old_root.resolve() and not allow_source_root:
        raise ValueError("Refusing original workspace; use --allow-source-root only intentionally")

    symlink_changes = []
    for entry in manifest.get("symlinks", []):
        link = workspace_path(root, entry["path"])
        if not link.is_symlink():
            raise ValueError(f"Missing symlink: {link}")
        filename = os.readlink(link)
        # Absolute symlinks can still resolve to the original machine during preflight.
        if Path(filename).is_absolute():
            source_target = Path(filename)
            if not source_target.is_relative_to(old_root):
                raise ValueError(f"Absolute symlink target outside original workspace: {filename}")
            target = root / source_target.relative_to(old_root)
            if not target.exists() or not target.resolve().is_relative_to(root):
                raise ValueError(f"Missing or external restored symlink target: {target}")
            symlink_changes.append((link, os.path.relpath(target, link.parent)))
        else:
            restored_target(filename, link.parent, old_root, root)

    hdf_files = [workspace_path(root, entry["path"]) for entry in manifest.get("files", [])
                 if entry.get("destination") != "skip"
                 and Path(entry["path"]).suffix.lower() in {".h5", ".hdf5"}]
    changes = defaultdict(list)
    targets = defaultdict(set)
    external_count = 0
    if hdf_files:
        import h5py
        for path in hdf_files:
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"Manifest HDF5 entry must be a restored regular file: {path}")
            with h5py.File(path, "r") as handle:
                names = []

                def collect(name, info):
                    if info.type == h5py.h5l.TYPE_EXTERNAL:
                        names.append(name)

                # HDF5 visits hard-linked groups only; external and soft links are leaves.
                handle.id.links.visit(collect, info=True)
                for name in names:
                    filename, dataset = handle.id.links.get_val(name)
                    filename, dataset = filename.decode(), dataset.decode()
                    target = restored_target(filename, path.parent, old_root, root)
                    targets[target].add(dataset)
                    external_count += 1
                    if Path(filename).is_absolute():
                        changes[path].append((name.decode(), os.path.relpath(target, path.parent), dataset))

        for target, datasets in targets.items():
            with h5py.File(target, "r") as handle:
                for dataset in datasets:
                    if not handle.id.links.exists(dataset.encode()):
                        raise ValueError(f"Missing external HDF5 dataset: {target}:{dataset}")

    # Complete every preflight check before modifying the restored copy.
    for link, filename in symlink_changes:
        link.unlink()
        link.symlink_to(filename)
    for path, links in changes.items():
        with h5py.File(path, "r+") as handle:
            for name, filename, dataset in links:
                del handle[name]
                handle[name] = h5py.ExternalLink(filename, dataset)
            handle.flush()

    return {"root": str(root), "symlinks_repaired": len(symlink_changes),
            "hdf5_files_checked": len(hdf_files), "external_links_checked": external_count,
            "external_links_repaired": sum(map(len, changes.values()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Restored workspace directory")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--allow-source-root", action="store_true",
                        help="Explicitly permit modification of the original workspace")
    args = parser.parse_args()
    try:
        result = repair(args.root, json.loads(args.manifest.read_text()), args.allow_source_root)
    except (ValueError, OSError, ImportError, KeyError, RuntimeError) as error:
        print(f"Link restoration failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
