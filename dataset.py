import os
import shutil
import kagglehub

print("donwloading")
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")

data_dir = os.path.join(os.getcwd(), "data")
os.makedirs(data_dir, exist_ok=True)

print("Moving files to /data")
for item in os.listdir(path):
    src = os.path.join(path, item)
    dst = os.path.join(data_dir, item)
    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)

print(f"Dataset available in: {data_dir}")
