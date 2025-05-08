import pandas as pd
from pathlib import Path
from PIL import Image

# map common PIL modes to bit depth (bits per pixel)
bit_depth_map = {
    '1': 1,
    'L': 8,
    'P': 8,
    'RGB': 24,
    'RGBA': 32,
    'CMYK': 32,
    'I;16': 16,
    'I': 32
}

# ←–– adjust this to your dataset root
root_dir = Path(r'C:\Users\matt\iCloudDrive\Family\Education\Matt\KBS\Modules\2025 T1\TECH3300 - Machine Learning Applications\Assessment 2\Assessment 2 Data')

records = []
for split in ("train", "test"):
    for cls_dir in (root_dir / split).iterdir():
        if not cls_dir.is_dir():
            continue
        for img_path in cls_dir.iterdir():
            if not img_path.is_file():
                continue

            size_bytes = img_path.stat().st_size
            with Image.open(img_path) as img:
                mode = img.mode
                bit_depth = bit_depth_map.get(mode, None)
                width, height = img.size
                # DPI tuple is often stored in 'dpi' metadata
                dpi = img.info.get('dpi', (None, None))
                h_res, v_res = dpi

            records.append({
                "Split": split,
                "Fruit": cls_dir.name,
                "FileName": img_path.name,
                "SizeBytes": size_bytes,
                "BitDepth": bit_depth,
                "Width": width,
                "Height": height,
                "HPPI": h_res,
                "VPPI": v_res
            })

df = pd.DataFrame(records)

# Summary statistics
print("\n=== File size (bytes) ===")
print(df["SizeBytes"].describe())

print("\n=== Bit depth counts ===")
print(df["BitDepth"].value_counts())

print("\n=== Image dimensions (px) counts ===")
print(df.groupby(["Width","Height"]).size().sort_values(ascending=False).head(10))

print("\n=== Resolution (HPPI × VPPI) counts ===")
print(df.groupby(["HPPI","VPPI"]).size().sort_values(ascending=False))

# (Optional) save to CSV for deeper analysis
df.to_csv(Path(__file__).parent/"image_properties.csv", index=False)
print(f"\nSaved full details to {Path(__file__).parent/'image_properties.csv'}")
