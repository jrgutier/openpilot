#!/usr/bin/env python3
"""Replace all instances of a string in file contents and filenames, preserving capitalization."""
import argparse
import os
import re
import sys

# Skip binary files and these directories
SKIP_DIRS = {".git", ".github", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache"}
BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg",
    ".mp4", ".mp3", ".wav", ".ogg",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z",
    ".bin", ".exe", ".dll", ".so", ".dylib", ".o", ".a",
    ".pyc", ".pyo", ".whl",
    ".pdf", ".doc", ".docx",
    ".ttf", ".otf", ".woff", ".woff2",
    ".pkl", ".onnx", ".dlc", ".thneed",
}


def build_variants(before: str, after: str) -> list[tuple[str, str]]:
    """Build case variants sorted longest-first to avoid partial replacements."""
    variants = [
        (before.lower(), after.lower()),           # openpilot -> sunnypilot
        (before.upper(), after.upper()),           # OPENPILOT -> SUNNYPILOT
        (before.capitalize(), after.capitalize()), # Openpilot -> Sunnypilot
        (before, after),                           # original as-is
    ]
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for b, a in variants:
        if b not in seen:
            seen.add(b)
            unique.append((b, a))
    # Sort longest first
    unique.sort(key=lambda x: len(x[0]), reverse=True)
    return unique


def build_pattern(variants: list[tuple[str, str]]) -> re.Pattern:
    """Compile a single regex that matches any variant."""
    alts = "|".join(re.escape(b) for b, _ in variants)
    return re.compile(f"({alts})")


def replace_contents(text: str, pattern: re.Pattern, lookup: dict[str, str]) -> str:
    """Replace all variant matches in text using the lookup table."""
    return pattern.sub(lambda m: lookup[m.group(0)], text)


def is_binary(filepath: str) -> bool:
    _, ext = os.path.splitext(filepath)
    if ext.lower() in BINARY_EXTENSIONS:
        return True
    try:
        with open(filepath, "rb") as f:
            chunk = f.read(8192)
            return b"\x00" in chunk
    except (OSError, PermissionError):
        return True


def process_directory(root_dir: str, before: str, after: str, dry_run: bool = False):
    variants = build_variants(before, after)
    lookup = {b: a for b, a in variants}
    pattern = build_pattern(variants)

    files_modified = 0
    files_renamed = 0
    renames = []  # collect (old_path, new_path) to do bottom-up

    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=True):
        # Prune skipped directories
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]

        for filename in filenames:
            filepath = os.path.join(dirpath, filename)

            if os.path.islink(filepath):
                continue

            # Replace file contents
            if not is_binary(filepath):
                try:
                    with open(filepath, "r", encoding="utf-8", errors="surrogateescape") as f:
                        content = f.read()
                except (OSError, PermissionError):
                    continue

                new_content = replace_contents(content, pattern, lookup)
                if new_content != content:
                    if dry_run:
                        print(f"  [content] {filepath}")
                    else:
                        with open(filepath, "w", encoding="utf-8", errors="surrogateescape") as f:
                            f.write(new_content)
                    files_modified += 1

            # Queue filename rename
            new_filename = replace_contents(filename, pattern, lookup)
            if new_filename != filename:
                renames.append((filepath, os.path.join(dirpath, new_filename)))

        # Queue directory renames
        for dirname in dirnames:
            new_dirname = replace_contents(dirname, pattern, lookup)
            if new_dirname != dirname:
                renames.append((
                    os.path.join(dirpath, dirname),
                    os.path.join(dirpath, new_dirname),
                ))

    # Rename bottom-up (deepest paths first) to avoid parent renames breaking child paths
    renames.sort(key=lambda x: x[0].count(os.sep), reverse=True)
    for old_path, new_path in renames:
        if os.path.exists(old_path):
            if dry_run:
                print(f"  [rename]  {old_path} -> {new_path}")
            else:
                os.rename(old_path, new_path)
                print(f"  [rename]  {old_path} -> {new_path}")
            files_renamed += 1

    return files_modified, files_renamed


def main():
    parser = argparse.ArgumentParser(description="Replace a string across an entire project (contents + filenames), preserving case.")
    parser.add_argument("folder", help="Root directory to process")
    parser.add_argument("before", help="String to find (e.g. openpilot)")
    parser.add_argument("after", help="String to replace with (e.g. sunnypilot)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change without modifying anything")
    args = parser.parse_args()

    root = os.path.abspath(args.folder)
    if not os.path.isdir(root):
        print(f"Error: {root} is not a directory", file=sys.stderr)
        sys.exit(1)

    mode = "DRY RUN" if args.dry_run else "LIVE"
    print(f"[{mode}] Replacing '{args.before}' -> '{args.after}' in {root}")
    print(f"  Variants:")
    for b, a in build_variants(args.before, args.after):
        print(f"    {b} -> {a}")
    print()

    modified, renamed = process_directory(root, args.before, args.after, dry_run=args.dry_run)
    print(f"\nDone. Files modified: {modified}, files/dirs renamed: {renamed}")


if __name__ == "__main__":
    main()
