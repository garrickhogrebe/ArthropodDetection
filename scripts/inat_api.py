"""
inat_api.py

This script interacts with the iNaturalist API to download images and associated metadata.
The images and metadata are organized in a directory structure that separates images by class and saves metadata in JSON format.
It is intended to be used for preparing datasets for machine learning classification tasks.

Usage:
    To download images, simply run the script. By default, it downloads images of bees, spiders, wasps, beetles, and moths.
    Modify the parameters as needed to adjust class names, taxon IDs, and the number of images to download.

Example:
    python inat_api.py
"""

from PIL import Image
import requests
import os
import json
import math

# Function to download bee images and save them in the appropriate folder structure
def download_images(class_name, taxon_id, output_folder="../output", num_images=200, per_page=50):
    """
    Downloads images and metadata for a specific class from the iNaturalist API.

    Args:
        class_name (str): Name of the class to be used in folder names (e.g., "bee").
        taxon_id (int): The iNaturalist taxon ID corresponding to the class.
        output_folder (str): Base folder to save downloaded images and metadata.
        num_images (int): Total number of images to download.
        per_page (int): Number of images to request per API call (limited by iNaturalist).

    The images are saved in "output_folder/images/class_name", and metadata is saved in "output_folder/metadata/class_name".
    """

    # Define folder paths
    jpg_folder      = os.path.join(output_folder, "images", class_name)
    metadata_folder = os.path.join(output_folder, "metadata", class_name)

    # Create directories if they don't exist
    os.makedirs(jpg_folder, exist_ok=True)
    os.makedirs(metadata_folder, exist_ok=True)
   
    num_pages  = math.ceil(num_images/per_page)
    page       = num_pages + 1
    
    while page > 1:

        page -= 1

        # iNaturalist API endpoint with research-grade filtering
        url = f'https://api.inaturalist.org/v1/observations?taxon_id={taxon_id}&quality_grade=research&per_page={per_page}&page={page}&identifications=most_agree'

        response = requests.get(url)
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            return

        data = response.json()

        # Download images, save metadata
        for i, result in enumerate(data['results']):
            if 'photos' in result:
                for photo in result['photos'][:1]:
                    # Get the image URL and download the JPEG
                    img_url       = photo['url'].replace('square', 'original')
                    img_data      = requests.get(img_url).content
                    img_file_name = f'{class_name}_{i + (page - 1) * per_page}'
                    
                    # Save JPEG image
                    jpg_file_path = os.path.join(jpg_folder, f'{img_file_name}.jpg')
                    with open(jpg_file_path, 'wb') as img_file:
                        img_file.write(img_data)

                    print(f"Downloaded JPEG: {jpg_file_path}")

                # Extract metadata and save it in the metadata folder
                metadata = {
                    "species_guess"  : result.get('species_guess', 'Unknown'),
                    "scientific_name": result['taxon']['name'],
                    "common_name"    : result['taxon'].get('common_name', {}).get('name', 'Unknown'),
                    "observed_on"    : result.get('observed_on', 'Unknown'),
                    "location"       : result.get('location', 'Unknown'),
                    "place_guess"    : result.get('place_guess', 'Unknown'),
                    "user"           : result['user']['login']
                }

                # Save metadata as JSON
                metadata_file_path = os.path.join(metadata_folder, f'{img_file_name}_metadata.json')
                with open(metadata_file_path, 'w') as metadata_file:
                    json.dump(metadata, metadata_file, indent=4)

                print(f"Saved metadata: {metadata_file_path}")

if __name__ == "__main__":
    num_images     = 10000
    output_folder = "../output"

    # Download images for each class
    download_images(class_name="bee",    output_folder=output_folder, taxon_id=630955, num_images=num_images)
    download_images(class_name="spider", output_folder=output_folder, taxon_id=47118,  num_images=num_images)
    download_images(class_name="wasp",   output_folder=output_folder, taxon_id=52747,  num_images=num_images)
    download_images(class_name="beetle", output_folder=output_folder, taxon_id=47208,  num_images=num_images)
    download_images(class_name="moth",   output_folder=output_folder, taxon_id=47157,  num_images=num_images)



