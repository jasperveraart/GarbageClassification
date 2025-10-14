import shutil
import pathlib

# Paths relative to this file
HERE = pathlib.Path(__file__).resolve().parent
RAW_ROOT = (HERE / ".." / "data" / "raw").resolve()
COMBINED_ROOT = (HERE / ".." / "data" / "combined").resolve()

def clean_dir(path: pathlib.Path):
    if not path.exists():
        print(f"[INFO] {path} does not exist, skipping.")
        return
    for item in path.iterdir():
        try:
            if item.is_dir() and not item.is_symlink():
                shutil.rmtree(item)
            else:
                item.unlink()
        except Exception as e:
            print(f"[WARN] Failed to delete {item}: {e}")
    print(f"[OK] Cleaned {path}")

def main():
    print("🧹 Removing all downloaded and merged dataset files...")
    clean_dir(RAW_ROOT)
    clean_dir(COMBINED_ROOT)
    print("\n✅ Cleanup complete. You can now rerun dataset_setup.py to rebuild everything.")

if __name__ == "__main__":
    main()
