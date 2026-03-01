"""Export named faces from an Adobe Lightroom Classic catalog (.lrcat).

Reads the SQLite catalog, crops face regions from source photos, and
optionally computes 512-d face encodings (via facenet-pytorch) for real-time matching.
"""

import collections
import json
import os
import re
import shutil
import sqlite3
import tempfile
import time
from pathlib import Path

from PIL import Image, ImageOps


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


def query_named_faces(conn, person_filter=None, confirmed_only=False):
    """Query all named, non-rejected faces from the catalog.

    Parameters
    ----------
    conn : sqlite3.Connection
    person_filter : str | None
        If set, only return faces for this exact person name.
    confirmed_only : bool
        If True, exclude suggested/unconfirmed faces (suggestedTag = 1).
        Only faces the user explicitly confirmed in Lightroom's People view
        will be returned.

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
    if confirmed_only:
        sql += "  AND (kwf.suggestedTag IS NULL OR kwf.suggestedTag != 1)\n"
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


def compute_encodings(output_dir, batch_size=64):
    """Walk person directories, compute 512-d face encodings, save as pickle.

    Uses facenet-pytorch (MTCNN + InceptionResnetV1) — no compilation required.
    Falls back to encoding the whole crop when MTCNN finds no face.
    Processes images in batches for significantly faster CPU/GPU throughput.
    """
    try:
        import torch
        from facenet_pytorch import MTCNN, InceptionResnetV1
    except ImportError:
        print("Error: facenet-pytorch not available. "
              "Install it with 'pip install facenet-pytorch' or use --skip-encodings.")
        return False

    import pickle
    import torchvision.transforms.functional as TF

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mtcnn = MTCNN(keep_all=False, image_size=160, device=device)
    model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    # Fallback transform: resize whole crop to 160x160 and normalize to [-1, 1]
    def fallback_tensor(img):
        t = TF.to_tensor(img.resize((160, 160)))
        return (t - 0.5) / 0.5

    output_dir = Path(output_dir)
    encodings_dict = {}
    total = 0

    # Collect all face files across all persons so we can count progress
    all_person_dirs = [d for d in sorted(output_dir.iterdir()) if d.is_dir()]
    all_files = [(d, f) for d in all_person_dirs for f in sorted(d.glob("face_*.jpg"))]
    n_total = len(all_files)
    print(f"Computing embeddings for {n_total} face images across "
          f"{len(all_person_dirs)} people (batch_size={batch_size})...")

    # MTCNN requires equal-dimension images for batch mode, but Lightroom crops
    # vary in size. So run MTCNN individually, then batch InceptionResnetV1
    # (all MTCNN outputs are 160x160, so embedding batching works fine).
    file_to_embedding = {}
    tensor_batch: list = []
    file_batch: list[str] = []

    def flush_embedding_batch():
        if not tensor_batch:
            return
        import torch as _torch
        batch_tensor = _torch.stack(tensor_batch).to(device)
        with _torch.no_grad():
            embeddings = model(batch_tensor).cpu().numpy()
        for fpath, emb in zip(file_batch, embeddings):
            file_to_embedding[fpath] = emb
        tensor_batch.clear()
        file_batch.clear()

    for i, (_, face_file) in enumerate(all_files):
        img = Image.open(str(face_file)).convert("RGB")

        ft = mtcnn(img)
        tensor_batch.append(ft if ft is not None else fallback_tensor(img))
        file_batch.append(str(face_file))

        if len(tensor_batch) >= batch_size:
            flush_embedding_batch()
            print(f"  {i + 1}/{n_total} images processed...", flush=True)

    flush_embedding_batch()
    print(f"  {n_total}/{n_total} images processed.", flush=True)

    # Reassemble per-person
    for person_dir in all_person_dirs:
        person_name = person_dir.name.replace('_', ' ')
        person_encodings = [
            file_to_embedding[str(f)]
            for f in sorted(person_dir.glob("face_*.jpg"))
            if str(f) in file_to_embedding
        ]
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
