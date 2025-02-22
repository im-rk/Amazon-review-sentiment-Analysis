! pip install kaggle
configuring the path of Kaggle.json file
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
import kagglehub
import os
import pandas as pd
import shutil

path = kagglehub.dataset_download("kritanjalijain/amazon-reviews")
print("Path to dataset files:", path)

files = os.listdir(path)
print("Files in dataset folder:", files)

csv_file = None
for file in files:
    if file.endswith(".csv"):
        csv_file = os.path.join(path, file)
        break

if csv_file:
    df = pd.read_csv(csv_file)
    print("CSV file loaded successfully!")
    print(df.head())  

    destination = "/content/amazon_reviews.csv"
    shutil.copy(csv_file, destination)
    print(f"File moved to: {destination}")

    df = pd.read_csv(destination)
else:
    print("No CSV file found in the dataset folder.")
