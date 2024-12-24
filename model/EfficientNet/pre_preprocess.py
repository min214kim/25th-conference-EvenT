import os
import shutil

# Define the directory containing the JSON files
source_directory = '/root/event/dataset/processed_val_labels'

# Define the output directory to store categorized files
output_directory = '/root/event/dataset/final_val_labels'

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Define the 4 specific categories to organize
categories = ["아우터", "하의", "상의", "원피스"]

# Process each file in the source directory
for filename in os.listdir(source_directory):
    if filename.endswith(".json"):  # Process only JSON files
        # Extract the clothing category from the filename
        try:
            # Example filename: "123_ClothesCategory.json"
            parts = filename.split("_")
            if len(parts) > 1:
                category = parts[1].split(".")[0]  # Extract category without extension
                
                if category in categories:  # Check if category is one of the specified
                    # Create a directory for this category if it doesn't exist
                    category_directory = os.path.join(output_directory, category)
                    os.makedirs(category_directory, exist_ok=True)
                    
                    # Move the file into the appropriate category directory
                    source_path = os.path.join(source_directory, filename)
                    destination_path = os.path.join(category_directory, filename)
                    shutil.move(source_path, destination_path)
                    print(f"Moved {filename} to {category_directory}")
                else:
                    print(f"Skipping {filename}, category {category} not in predefined list.")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

print("Files have been successfully categorized!")
