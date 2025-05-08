import csv
from pathlib import Path

# 1) Adjust this to point at your data folder
root_dir = Path(r'C:\Users\matt\iCloudDrive\Family\Education\Matt\KBS\Modules\2025 T1\TECH3300 - Machine Learning Applications\Assessment 2\Assessment 2 Data')

# 2) Prepare CSV file path
csv_path = Path(__file__).parent / 'data_summary.csv'

# 3) Gather counts
rows = []
for cls_dir in sorted((root_dir / "train").iterdir()):
    if not cls_dir.is_dir():
        continue
    fruit = cls_dir.name
    train_count = sum(1 for _ in (root_dir / "train" / fruit).iterdir() if _.is_file())
    test_count  = sum(1 for _ in (root_dir / "test"  / fruit).iterdir() if _.is_file())
    rows.append((fruit, train_count, test_count))

# 4) Write CSV
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Fruit', 'Train', 'Test'])
    writer.writerows(rows)

# 5) Print CSV to terminal
with open(csv_path, 'r') as f:
    print(f.read())
