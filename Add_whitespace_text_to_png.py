from PIL import Image, ImageDraw, ImageFont
import os

def add_white_space_with_banners(input_image_path, output_image_path, top_text, bottom_text, font_size=20):
    if input_image_path.endswith('.png'):
        # Open the image
        img = Image.open(input_image_path)

        # Get original dimensions and calculate new height
        width, height = img.size
        
        # Prepare for drawing text
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()

        # Calculate the height of the top and bottom text banners
        top_text_bbox = draw.textbbox((0, 0), top_text, font=font)
        top_text_height = top_text_bbox[3] - top_text_bbox[1]

        bottom_text_bbox = draw.textbbox((0, 0), bottom_text, font=font)
        bottom_text_height = bottom_text_bbox[3] - bottom_text_bbox[1]

        # Calculate new height to include space for both banners
        top_banner_height = top_text_height + 20  # Add padding
        bottom_banner_height = bottom_text_height + 20  # Add padding
        new_height = height + top_banner_height + bottom_banner_height  # space for both banners

        # Create a new white background image
        new_img = Image.new('RGB', (width, new_height), color=(255, 255, 255))

        # Paste the original image onto the new canvas, leaving space for banners
        new_img.paste(img, (0, top_banner_height))

        # Prepare for drawing text on the banners
        draw = ImageDraw.Draw(new_img)

        # Calculate and draw the top banner text
        top_text_position = (
            10,  # Positioned at left with padding
            (top_banner_height - top_text_height) // 2  # Vertically centered
        )
        draw.text(top_text_position, top_text, fill=(0, 0, 0), font=font)

        # Calculate and draw the bottom banner text
        bottom_text_position = (
            10,  # Positioned at left with padding
            new_height - bottom_banner_height + (bottom_banner_height - bottom_text_height) // 2  # Vertically centered
        )
        draw.text(bottom_text_position, bottom_text, fill=(0, 0, 0), font=font)

        # Save the new image with banners
        new_img.save(output_image_path)
        #print(f"Processed and saved image as {output_image_path}")
    else:
        print(f"Skipping adding text to {input_image_path} as it is not a PNG file")
