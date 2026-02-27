"""Export named faces from an Adobe Lightroom Classic catalog (.lrcat).

Reads the SQLite catalog, crops face regions from source photos, and
optionally computes 128-d face encodings for real-time matching.
"""

import argparse
import collections
import json
import os
import re
import shutil
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

from PIL import Image, ImageOps


def parse_args():
    p = argparse.ArgumentParser(
        description="Export named/confirmed faces from a Lightroom Classic catalog."
    )
    p.add_argument(
        "--catalog", required=True,
        help="Path to Lightroom .lrcat catalog file."
    )
    p.add_argument(
        "--output", default="./faces/",
        help="Output directory for cropped faces and encodings (default: ./faces/)."
    )
    p.add_argument(
        "--person", default=None,
        help="Export only this person (exact name match). Omit to export all."
    )
    p.add_argument(
        "--padding", type=int, default=40,
        help="Pixels of padding around each face crop (default: 40)."
    )
    p.add_argument(
        "--min-size", type=int, default=50,
        help="Skip face crops smaller than this in either dimension (default: 50)."
    )
    p.add_argument(
        "--skip-encodings", action="store_true",
        help="Skip computing face encodings (useful if dlib is not installed)."
    )
    p.add_argument(
        "--root-remap", action="append", default=[], metavar="SRC=>DST",
        help="Remap catalog root paths. E.g. --root-remap \"X:/=>/photos\" "
             "replaces paths starting with X:/ to /photos/. Can be repeated."
    )
    return p.parse_args()


def parse_root_remaps(remap_args):
    """Parse --root-remap arguments into (src, dst) tuples.

    Each argument has the form "SRC=>DST". Backslashes are normalized to
    forward slashes and trailing slashes are ensured for prefix matching.
    """
    remaps = []
    for arg in remap_args:
        if "=>" not in arg:
            raise ValueError(f"Invalid --root-remap format (expected SRC=>DST): {arg}")
        src, dst = arg.split("=>", 1)
        src = src.replace("\\", "/").rstrip("/") + "/"
        dst = dst.replace("\\", "/").rstrip("/") + "/"
        remaps.append((src, dst))
    return remaps


def open_catalog(catalog_path):
    """Copy the .lrcat file to a temp directory and open it read-only.

    Returns (connection, temp_dir_path). Caller must clean up temp_dir.
    """
    catalog_path = Path(catalog_path).resolve()
    if not catalog_path.exists():
        raise FileNotFoundError(f"Catalog not found: {catalog_path}")
    if not catalog_path.suffix.lower() == ".lrcat":
        raise ValueError(f"Expected a .lrcat file, got: {catalog_path.suffix}")

    temp_dir = tempfile.mkdtemp(prefix="lr_faces_")
    temp_catalog = Path(temp_dir) / catalog_path.name
    print(f"Copying catalog to temp: {temp_catalog}")
    shutil.copy2(catalog_path, temp_catalog)

    # Also copy the -wal and -shm files if they exist (for WAL mode catalogs)
    for suffix in ["-wal", "-shm"]:
        wal_file = catalog_path.parent / (catalog_path.name + suffix)
        if wal_file.exists():
            shutil.copy2(wal_file, Path(temp_dir) / (catalog_path.name + suffix))

    conn = sqlite3.connect(f"file:{temp_catalog}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    # Validate that the expected tables exist
    tables = {row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    required = {"AgLibraryFace", "AgLibraryKeywordFace", "AgLibraryKeyword",
                "Adobe_images", "AgLibraryFile", "AgLibraryFolder",
                "AgLibraryRootFolder"}
    missing = required - tables
    if missing:
        conn.close()
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError(
            f"Catalog is missing expected tables: {', '.join(sorted(missing))}. "
            "Is this a Lightroom Classic catalog?"
        )

    return conn, temp_dir


def query_named_faces(conn, person_filter=None):
    """Query all named, non-rejected faces from the catalog.

    Returns a list of dicts with keys:
        name, face_id, tl_x, tl_y, br_x, br_y,
        root_path, path_from_root, base_name, extension
    """
    sql = """
        SELECT
            kw.name           AS person_name,
            face.id_local     AS face_id,
            face.tl_x         AS tl_x,
            face.tl_y         AS tl_y,
            face.br_x         AS br_x,
            face.br_y         AS br_y,
            root.absolutePath  AS root_path,
            folder.pathFromRoot AS path_from_root,
            file.baseName      AS base_name,
            file.extension     AS extension
        FROM AgLibraryFace face
        JOIN AgLibraryKeywordFace kwf ON kwf.face = face.id_local
        JOIN AgLibraryKeyword kw ON kw.id_local = kwf.tag
        JOIN Adobe_images img ON img.id_local = face.image
        JOIN AgLibraryFile file ON file.id_local = img.rootFile
        JOIN AgLibraryFolder folder ON folder.id_local = file.folder
        JOIN AgLibraryRootFolder root ON root.id_local = folder.rootFolder
        WHERE kw.name IS NOT NULL
          AND kw.name != ''
          AND (kwf.userReject IS NULL OR kwf.userReject != 1)
    """
    params = []
    if person_filter:
        sql += " AND kw.name = ?"
        params.append(person_filter)

    sql += " ORDER BY kw.name, face.id_local"

    rows = conn.execute(sql, params).fetchall()
    results = []
    for row in rows:
        results.append({
            "name": row["person_name"],
            "face_id": row["face_id"],
            "tl_x": row["tl_x"],
            "tl_y": row["tl_y"],
            "br_x": row["br_x"],
            "br_y": row["br_y"],
            "root_path": row["root_path"],
            "path_from_root": row["path_from_root"],
            "base_name": row["base_name"],
            "extension": row["extension"],
        })
    return results


def resolve_image_path(root_path, path_from_root, base_name, extension,
                       root_remaps=None):
    """Build a full filesystem path from Lightroom's decomposed path components."""
    # Lightroom stores root as e.g. "/Users/name/Pictures/" or "C:/Photos/"
    # path_from_root is relative folder, base_name + extension is the file
    if root_remaps:
        normalized = root_path.replace("\\", "/")
        if not normalized.endswith("/"):
            normalized += "/"
        for src, dst in root_remaps:
            if normalized.startswith(src):
                root_path = dst + normalized[len(src):]
                break

    filename = f"{base_name}.{extension}" if extension else base_name
    full = Path(root_path) / path_from_root / filename
    return full


def load_image_with_orientation(path):
    """Load an image with EXIF orientation applied.

    Returns a PIL RGB Image, or None if the file cannot be loaded.
    """
    try:
        pil_img = Image.open(path)
        pil_img = ImageOps.exif_transpose(pil_img)
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        return pil_img
    except Exception:
        return None


def crop_face(image, tl_x, tl_y, br_x, br_y, padding=40):
    """Crop a face region from a PIL Image using normalized coordinates.

    Lightroom stores face coordinates as normalized floats (0.0-1.0)
    relative to the EXIF-oriented image dimensions.

    Returns the cropped PIL Image, or None if the crop is invalid.
    """
    w, h = image.size  # PIL uses (width, height)

    # Convert normalized coords to pixel coords
    x1 = int(tl_x * w)
    y1 = int(tl_y * h)
    x2 = int(br_x * w)
    y2 = int(br_y * h)

    # Apply padding
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)

    if x2 <= x1 or y2 <= y1:
        return None

    return image.crop((x1, y1, x2, y2))


def sanitize_name(name):
    """Convert a person name to a filesystem-safe directory name."""
    # Replace spaces with underscores, remove unsafe characters
    safe = re.sub(r'[<>:"/\\|?*]', '', name)
    safe = safe.replace(' ', '_')
    safe = safe.strip('._')
    return safe or "Unknown"


def compute_encodings(output_dir):
    """Walk person directories, compute 128-d face encodings, save as pickle.

    Requires the face_recognition library (and dlib).
    """
    try:
        import face_recognition
    except ImportError:
        print("Error: face_recognition library not available. "
              "Install it or use --skip-encodings.")
        return False

    import pickle

    output_dir = Path(output_dir)
    encodings_dict = {}
    total = 0

    for person_dir in sorted(output_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        person_name = person_dir.name.replace('_', ' ')
        person_encodings = []

        face_files = sorted(person_dir.glob("face_*.jpg"))
        for face_file in face_files:
            img = face_recognition.load_image_file(str(face_file))
            encs = face_recognition.face_encodings(img)
            if encs:
                person_encodings.append(encs[0])
            else:
                # Use the whole crop as the face if detection fails
                encs = face_recognition.face_encodings(img, known_face_locations=[
                    (0, img.shape[1], img.shape[0], 0)  # top, right, bottom, left
                ])
                if encs:
                    person_encodings.append(encs[0])

        if person_encodings:
            encodings_dict[person_name] = person_encodings
            total += len(person_encodings)
            print(f"  {person_name}: {len(person_encodings)} encodings")

    pkl_path = output_dir / "encodings.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(encodings_dict, f)

    print(f"Saved {total} encodings for {len(encodings_dict)} people to {pkl_path}")
    return True


def save_export_log(output_dir, stats):
    """Write a JSON log with export statistics."""
    log_path = Path(output_dir) / "export_log.json"
    stats["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Export log saved to {log_path}")


def main():
    args = parse_args()
    root_remaps = parse_root_remaps(args.root_remap)
    catalog_path = args.catalog
    output_dir = Path(args.output)
    temp_dir = None

    try:
        # Open catalog safely
        print(f"Opening catalog: {catalog_path}")
        conn, temp_dir = open_catalog(catalog_path)

        # Query faces
        print("Querying named faces...")
        faces = query_named_faces(conn, args.person)
        print(f"Found {len(faces)} face regions")

        if not faces:
            if args.person:
                print(f"No faces found for person '{args.person}'.")
            else:
                print("No named faces found in catalog.")
            conn.close()
            return

        # Prepare output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Track stats
        stats = {
            "total_faces_in_catalog": len(faces),
            "exported": 0,
            "skipped_missing_image": 0,
            "skipped_load_failed": 0,
            "skipped_too_small": 0,
            "skipped_crop_failed": 0,
            "per_person": {},
            "warnings": [],
        }

        # Group by person for sequential numbering
        person_counters = {}
        # LRU cache: keep at most 100 images in memory to avoid OOM
        IMAGE_CACHE_SIZE = 100
        image_cache = collections.OrderedDict()
        failed_paths = set()  # Paths that failed to load (don't retry)

        for i, face in enumerate(faces):
            name = face["name"]
            safe_name = sanitize_name(name)

            # Resolve source image path
            img_path = resolve_image_path(
                face["root_path"], face["path_from_root"],
                face["base_name"], face["extension"],
                root_remaps=root_remaps,
            )

            if not img_path.exists():
                stats["skipped_missing_image"] += 1
                continue

            img_key = str(img_path)

            # Skip paths we already know can't be loaded
            if img_key in failed_paths:
                stats["skipped_load_failed"] += 1
                continue

            # Load image (with LRU caching)
            if img_key in image_cache:
                image_cache.move_to_end(img_key)
                image = image_cache[img_key]
            else:
                image = load_image_with_orientation(img_path)
                if image is None:
                    failed_paths.add(img_key)
                    stats["skipped_load_failed"] += 1
                    continue
                image_cache[img_key] = image
                if len(image_cache) > IMAGE_CACHE_SIZE:
                    image_cache.popitem(last=False)

            # Crop face
            crop = crop_face(
                image, face["tl_x"], face["tl_y"],
                face["br_x"], face["br_y"], args.padding
            )

            if crop is None:
                stats["skipped_crop_failed"] += 1
                continue

            # Check minimum size
            cw, ch = crop.size  # PIL uses (width, height)
            if ch < args.min_size or cw < args.min_size:
                stats["skipped_too_small"] += 1
                continue

            # Save crop
            person_dir = output_dir / safe_name
            person_dir.mkdir(exist_ok=True)

            counter = person_counters.get(safe_name, 0) + 1
            person_counters[safe_name] = counter

            face_path = person_dir / f"face_{counter:03d}.jpg"
            crop.save(str(face_path), "JPEG", quality=95)

            stats["exported"] += 1
            stats["per_person"][name] = stats["per_person"].get(name, 0) + 1

            # Progress
            if (i + 1) % 100 == 0 or (i + 1) == len(faces):
                print(f"  Processed {i + 1}/{len(faces)} faces "
                      f"({stats['exported']} exported, "
                      f"{stats['skipped_load_failed']} unsupported format)...")

        image_cache.clear()

        conn.close()

        print(f"\nExported {stats['exported']} faces to {output_dir}")
        if stats["skipped_missing_image"]:
            print(f"  Skipped {stats['skipped_missing_image']} (missing source image)")
        if stats["skipped_load_failed"]:
            print(f"  Skipped {stats['skipped_load_failed']} (image load failed)")
        if stats["skipped_too_small"]:
            print(f"  Skipped {stats['skipped_too_small']} (crop too small)")
        if stats["skipped_crop_failed"]:
            print(f"  Skipped {stats['skipped_crop_failed']} (invalid crop region)")

        # Save export log
        save_export_log(output_dir, stats)

        # Compute encodings unless skipped
        if not args.skip_encodings:
            print("\nComputing face encodings...")
            compute_encodings(output_dir)
        else:
            print("\nSkipping face encoding computation (--skip-encodings).")

    finally:
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
