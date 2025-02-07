! pip install kaggle
configuring the path of Kaggle.json file
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
import kagglehub
import os
import pandas as pd
import shutil

# Step 1: Download dataset
path = kagglehub.dataset_download("kritanjalijain/amazon-reviews")
print("Path to dataset files:", path)

# Step 2: List files in the dataset folder
files = os.listdir(path)
print("Files in dataset folder:", files)

# Step 3: Find the CSV file (modify if needed)
csv_file = None
for file in files:
    if file.endswith(".csv"):
        csv_file = os.path.join(path, file)
        break

# Step 4: If CSV is found, load it into Pandas
if csv_file:
    df = pd.read_csv(csv_file)
    print("CSV file loaded successfully!")
    print(df.head())  # Show first few rows

    # Step 5: Move the CSV to Google Colab's /content/ directory for easy access
    destination = "/content/amazon_reviews.csv"
    shutil.copy(csv_file, destination)
    print(f"File moved to: {destination}")

    # Now you can load it directly with:
    df = pd.read_csv(destination)
else:
    print("No CSV file found in the dataset folder.")
