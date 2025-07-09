import os
import pandas as pd

# Go up one directory to access the 'train' folder
data_dir = "../train"
records = []

for label in ["NORMAL", "PNEUMONIA"]:
    folder = os.path.join(data_dir, label)
    if not os.path.exists(folder):
        print(f"❌ Folder not found: {folder}")
        continue
    for filename in os.listdir(folder):
        if filename.lower().endswith((".jpeg", ".jpg", ".png")):
            records.append({"filename": filename, "label": label})

df = pd.DataFrame(records)
df.to_csv("train_data.csv", index=False)
print("✅ CSV created with", len(df), "records.")
