import os
import shutil

# Paths to the source directories and the output directory
images_dir = '/root/25th-conference-EvenT/dataset/EfficientNet_data/true_train_images'
jsons_dir = '/root/25th-conference-EvenT/dataset/EfficientNet_data/true_train_labels'
output_dir = '/root/25th-conference-EvenT/dataset/EfficientNet_data/train'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Iterate through the subdirectories in the images directory
for subdir in os.listdir(images_dir):
    image_subdir_path = os.path.join(images_dir, subdir)
    json_subdir_path = os.path.join(jsons_dir, subdir)
    output_subdir_path = os.path.join(output_dir, subdir)

    # Ensure the subdirectory is a directory and exists in both sources
    if os.path.isdir(image_subdir_path) and os.path.isdir(json_subdir_path):
        # Create the corresponding subdirectory in the output directory
        os.makedirs(output_subdir_path, exist_ok=True)

        # Copy all images from the image subdirectory
        for file in os.listdir(image_subdir_path):
            image_file_path = os.path.join(image_subdir_path, file)
            if os.path.isfile(image_file_path):
                shutil.copy(image_file_path, output_subdir_path)

        # Copy all JSON files from the JSON subdirectory
        for file in os.listdir(json_subdir_path):
            json_file_path = os.path.join(json_subdir_path, file)
            if os.path.isfile(json_file_path):
                shutil.copy(json_file_path, output_subdir_path)

        print(f"Copied files to {output_subdir_path}")

print("All files have been merged into the output directory.")
