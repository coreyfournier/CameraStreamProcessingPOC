"""CLI entry point for exporting named faces from a Lightroom Classic catalog."""

import argparse
import shutil
from pathlib import Path

from application.catalog_exporter import (
    open_catalog,
    query_named_faces,
    resolve_image_path,
    load_image_with_orientation,
    crop_face,
    sanitize_name,
    compute_encodings,
    save_export_log,
)
import collections


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
        help="Skip computing face encodings (requires facenet-pytorch)."
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
            "skipped_already_exists": 0,
            "skipped_missing_image": 0,
            "skipped_load_failed": 0,
            "skipped_too_small": 0,
            "skipped_crop_failed": 0,
            "per_person": {},
            "warnings": [],
        }

        # LRU cache: keep at most 100 images in memory to avoid OOM
        IMAGE_CACHE_SIZE = 100
        image_cache = collections.OrderedDict()
        failed_paths = set()  # Paths that failed to load (don't retry)

        for i, face in enumerate(faces):
            name = face["name"]
            safe_name = sanitize_name(name)
            face_id = face["face_id"]

            # Check if already exported (face_id-based filename)
            person_dir = output_dir / safe_name
            face_path = person_dir / f"face_{face_id}.jpg"
            if face_path.exists():
                stats["skipped_already_exists"] += 1
                stats["per_person"][name] = stats["per_person"].get(name, 0) + 1
                continue

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
            person_dir.mkdir(exist_ok=True)
            face_path = person_dir / f"face_{face_id}.jpg"
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
        if stats["skipped_already_exists"]:
            print(f"  Skipped {stats['skipped_already_exists']} (already exported)")
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
