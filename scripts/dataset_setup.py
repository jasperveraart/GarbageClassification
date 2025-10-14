import os
import pathlib
import shutil
import hashlib
from collections import Counter, defaultdict

import kagglehub  # pip install kagglehub

# final classes
FINAL_CLASSES = ["glass", "metal", "paper", "plastic", "residual"]

# copy files (True) or make hard links to save space (False, same disk only)
COPY_FILES = True

# skip exact duplicates by content hash
SKIP_DUPLICATES = True

# include image types
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# datasets to fetch
DATASETS = [
    "alistairking/recyclable-and-household-waste-classification",
    "asdasdasasdas/garbage-classification",
    "farzadnekouei/trash-type-image-dataset",
]

# where to place data relative to this file
HERE = pathlib.Path(__file__).resolve().parent
RAW_ROOT = (HERE / ".." / "data" / "raw").resolve()
OUT_ROOT = (HERE / ".." / "data" / "combined").resolve()


def norm_key(name: str) -> str:
    n = name.strip().lower().replace("_", " ").replace("-", " ")
    n = " ".join(n.split())
    return n


# map many source names to your five classes
# use None to exclude from the dataset
ALIAS = {
    # glass
    "glass": "glass",
    "glass bottle": "glass",
    "glass bottles": "glass",
    "glass beverage bottles": "glass",
    "glass cosmetic containers": "glass",
    "glass food jars": "glass",
    "glass jar": "glass",
    "glass jars": "glass",

    # metal
    "metal": "metal",
    "can": "metal",
    "cans": "metal",
    "tin can": "metal",
    "tin cans": "metal",
    "aerosol cans": "metal",
    "aluminum can": "metal",
    "aluminum cans": "metal",
    "aluminum soda cans": "metal",
    "aluminum food cans": "metal",
    "aluminium can": "metal",
    "aluminium cans": "metal",
    "aluminium soda cans": "metal",
    "aluminium food cans": "metal",
    "scrap metal": "metal",
    "metal container": "metal",

    # paper
    "paper": "paper",
    "papers": "paper",
    "office paper": "paper",
    "magazine": "paper",
    "magazines": "paper",
    "newspaper": "paper",
    "newspapers": "paper",
    "paper cup": "paper",
    "paper cups": "paper",
    "cardboard": "paper",
    "cardboard box": "paper",
    "cardboard boxes": "paper",
    "cardboard packaging": "paper",
    "paper and cardboard": "paper",
    "paper cardboard": "paper",

    # plastic
    "plastic": "plastic",
    "plastics": "plastic",
    "plastic bottle": "plastic",
    "plastic bottles": "plastic",
    "pet bottle": "plastic",
    "pet bottles": "plastic",
    "plastic cup lids": "plastic",
    "plastic detergent bottles": "plastic",
    "plastic food containers": "plastic",
    "plastic shopping bags": "plastic",
    "disposable plastic cutlery": "plastic",
    "plastic container": "plastic",
    "plastic containers": "plastic",

    # residual
    "trash": "residual",
    "other": "residual",
    "garbage": "residual",
    "residual": "residual",
    "waste": "residual",
    "non recyclable": "residual",
    "coffee grounds": "residual",
    "eggshells": "residual",
    "food waste": "residual",
    "biodegradable": "residual",
    "organic": "residual",
    "biological": "residual",

    # exclude items you do not want
    "clothes": None,
    "clothing": None,
    "textile": None,
    "cloth": None,
    "battery": None,
    "batteries": None,
}


def is_image(path: pathlib.Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def ensure_out_dirs():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for c in FINAL_CLASSES:
        (OUT_ROOT / c).mkdir(parents=True, exist_ok=True)


def walk_leaf_dirs(root: pathlib.Path):
    for cur, dirs, files in os.walk(root):
        if not dirs:
            yield pathlib.Path(cur), files


def sha1_of_file(p: pathlib.Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    # short and readable id
    return h.hexdigest()[:10]


def destination_for(label: str, src: pathlib.Path, digest: str) -> pathlib.Path:
    ext = src.suffix.lower()
    # class prefix and stable hash
    name = f"{label}_{digest}{ext}"
    return (OUT_ROOT / label / name)


def download_all():
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    local_roots = []
    for ds in DATASETS:
        print(f"Downloading {ds} with kagglehub")
        path = pathlib.Path(kagglehub.dataset_download(ds))
        # make a readable link in data raw to the cached version
        # cached path looks like ~/.cache/kagglehub/datasets/<owner>/<dataset>/versions/<ver>
        owner = path.parts[-4]
        dsname = path.parts[-3]
        ver = path.parts[-1]
        marker = RAW_ROOT / f"{owner}_{dsname}_v{ver}"
        if not marker.exists():
            marker.symlink_to(path, target_is_directory=True)
        local_roots.append(path)
        print("  ->", path)
    return local_roots


def route_name(name: str):
    key = norm_key(name)
    return ALIAS.get(key)  # can be a class name or None


def merge_all(local_roots):
    ensure_out_dirs()

    written = 0
    skipped_excluded = 0
    skipped_unknown = 0
    skipped_duplicates = 0

    per_class = Counter()
    per_source = defaultdict(int)
    unknown_dirs = set()
    seen_hashes = set()

    for root in local_roots:
        for leaf_dir, files in walk_leaf_dirs(root):
            if not files:
                continue

            # try leaf dir name, then parent dir name
            label = route_name(leaf_dir.name)
            if label is None:
                # None means explicit exclude, skip without logging unknown
                # try parent only if leaf was not an explicit exclude
                parent_label = route_name(leaf_dir.parent.name)
                label = parent_label

            # handle explicit exclusion
            if label is None:
                skipped_excluded += len(files)
                continue

            # if label not mapped into final classes, consider unknown
            if label not in FINAL_CLASSES:
                # collect unknown only if neither leaf nor parent mapped
                if route_name(leaf_dir.name) is None and route_name(leaf_dir.parent.name) is None:
                    unknown_dirs.add(str(leaf_dir))
                    skipped_unknown += len(files)
                continue

            for fname in files:
                src = leaf_dir / fname
                if not src.is_file() or not is_image(src):
                    continue

                digest = sha1_of_file(src)

                if SKIP_DUPLICATES and digest in seen_hashes:
                    skipped_duplicates += 1
                    continue
                seen_hashes.add(digest)

                dst = destination_for(label, src, digest)

                if COPY_FILES:
                    shutil.copy2(src, dst)
                else:
                    # hard link to save space if same filesystem
                    try:
                        os.link(src, dst)
                    except OSError:
                        shutil.copy2(src, dst)

                written += 1
                per_class[label] += 1
                per_source[str(root)] += 1

    print("\nDone building combined dataset")
    print("Images written:", written)
    print("Skipped (excluded):", skipped_excluded)
    print("Skipped (unknown folders):", skipped_unknown)
    print("Skipped (duplicates):", skipped_duplicates)

    print("\nPer class counts:")
    for c in FINAL_CLASSES:
        print(f"  {c}: {per_class[c]}")

    print("\nPer source counts:")
    for src, n in per_source.items():
        print(f"  {src}: {n}")

    if unknown_dirs:
        print("\nUnmapped folders, add to ALIAS if they contain valid images:")
        for d in sorted(list(unknown_dirs))[:80]:
            print(" ", d)
        if len(unknown_dirs) > 80:
            print("  ...")


def main():
    print("Step 1. Downloading datasets")
    roots = download_all()
    print("\nStep 2. Merging into", OUT_ROOT)
    merge_all(roots)
    print("\nAll set. Point your training code to:")
    print("DATASET_DIR =", str(OUT_ROOT))


if __name__ == "__main__":
    main()
