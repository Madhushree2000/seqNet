import os
import glob
from pathlib import Path

def generate_image_list(folder_path, output_file="image_list.txt"):
    """
    Creates a text file containing the names of all image files in the specified folder.
    
    Args:
        folder_path (str): Path to the folder containing images
        output_file (str): Name of the output text file
    """
    # Path to the folder containing the images
    image_folder = folder_path
    
    # Find all PNG files in the folder
    image_pattern = os.path.join(image_folder, "frame_*.png")
    image_files = glob.glob(image_pattern)
    
    # If no files found, try with standard image extensions
    if not image_files:
        print("No 'frame_*.png' files found. Trying with standard image extensions...")
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff', '*.webp']
        for extension in image_extensions:
            file_pattern = os.path.join(image_folder, extension)
            image_files.extend(glob.glob(file_pattern))
    
    # Sort the files numerically (frame_0.png, frame_1.png, ...)
    image_files.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    
    # Get only the filenames without the full path
    image_names = [Path(file).name for file in image_files]
    
    # Write the filenames to a text file, one per line
    with open(output_file, 'w') as f:
        for name in image_names:
            f.write(f"{name}\n")
    
    print(f"Successfully created {output_file} with {len(image_names)} image files.")
    return len(image_names)

if __name__ == "__main__":
    # Your specific folder path
    folder_path = "/Users/madhushreesannigrahi/Documents/GitHub/seqNet/data/underwater_data/Traj_5_frames/cam0"
    output_file = "image_list.txt"
    
    # Check if the folder exists
    if not os.path.isdir(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist.")
    else:
        # Generate the image list
        num_images = generate_image_list(folder_path, output_file)
        
        # Debug information
        if num_images == 0:
            print("No images were found in the folder. Let's check what files are actually there:")
            all_files = os.listdir(folder_path)
            print(f"Files in directory: {all_files[:10]}{'...' if len(all_files) > 10 else ''}")