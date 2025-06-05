import os
import re
from PIL import Image

def crop_images_in_folder(folder_path, regex_pattern, crop_width, crop_height):
    """
    Crops all images in a folder that match a regex pattern.

    The crop is centered, and the cropped images are saved in a new
    subfolder named 'crops_{width}_{height}'.

    Args:
        folder_path (str): The path to the folder containing the images.
        regex_pattern (str): The regular expression to match filenames against.
        crop_width (int): The desired width of the cropped image.
        crop_height (int): The desired height of the cropped image.
    """
    if not os.path.isdir(folder_path):
        print(f"Error: Folder not found at '{folder_path}'")
        return

    output_folder_name = f"crops_{crop_width}_{crop_height}"
    output_folder_path = os.path.join(folder_path, output_folder_name)

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
        print(f"Created output folder: '{output_folder_path}'")

    processed_count = 0
    skipped_count = 0

    for filename in os.listdir(folder_path):
        if re.match(regex_pattern, filename):
            image_path = os.path.join(folder_path, filename)
            try:
                with Image.open(image_path) as img:
                    original_width, original_height = img.size

                    if crop_width > original_width or crop_height > original_height:
                        print(f"Skipping '{filename}': Crop dimensions ({crop_width}x{crop_height}) "
                              f"are larger than image dimensions ({original_width}x{original_height}).")
                        skipped_count += 1
                        continue

                    # Calculate coordinates for a centered crop
                    left = (original_width - crop_width) / 2
                    top = (original_height - crop_height) / 2
                    right = (original_width + crop_width) / 2
                    bottom = (original_height + crop_height) / 2

                    # Perform the crop
                    cropped_img = img.crop((left, top, right, bottom))

                    # Save the cropped image
                    output_image_path = os.path.join(output_folder_path, filename)
                    cropped_img.save(output_image_path)
                    print(f"Cropped '{filename}' and saved to '{output_image_path}'")
                    processed_count += 1

            except FileNotFoundError:
                print(f"Error: Image file not found at '{image_path}' (should not happen if os.listdir is used).")
                skipped_count += 1
            except Exception as e:
                print(f"Error processing '{filename}': {e}")
                skipped_count += 1
        else:
            # If the file doesn't match the regex and it's not a directory
            if os.path.isfile(os.path.join(folder_path, filename)) and filename != output_folder_name :
                # You might want to log these or handle them differently
                pass


    if processed_count > 0:
        print(f"\nSuccessfully processed and cropped {processed_count} image(s).")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} image(s) due to errors or size constraints.")
    if processed_count == 0 and skipped_count == 0:
        print("No images matching the regex pattern were found in the specified folder.")

if __name__ == "__main__":
    # --- Configuration ---
    # ⚠️ IMPORTANT: Replace these with your actual values!
    INPUT_FOLDER = r"C:\Users\obs\Desktop\Diameter estimate variations\segmented"  # e.g., "/Users/yourname/Pictures/MyPhotos"
    # Regex to match, e.g., r".*\.jpg$" for all JPGs, or r"image_\d{3}\.png$" for image_001.png etc.
    REGEX = r"\d+-\d+\.png$" # Matches common image extensions (case-insensitive due to re.IGNORECASE not used here, but PIL handles various cases)
    CROP_W = 0.75*1000  # Desired width of the crop
    CROP_H = 0.75*1000  # Desired height of the crop

    if INPUT_FOLDER == "your_image_folder_path":
        print("Please update the 'INPUT_FOLDER' variable in the script with the actual path to your images.")
    else:
        crop_images_in_folder(INPUT_FOLDER, REGEX, CROP_W, CROP_H)