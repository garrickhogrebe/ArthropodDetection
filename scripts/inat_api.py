"""
Script to download images from iNaturalist website

Modify main below to select which taxon id to download images of

Outputs organized into:

"""

from PIL import Image
import requests
import os
import json
import math

# Function to download bee images and save them in the appropriate folder structure
def download_images(class_name, taxon_id, num_images=200, per_page=50):
    # Define folder paths
    jpg_folder      = f'../output/images/{class_name}'
    metadata_folder = f'../output/metadata/{class_name}'

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

        # Download images and convert to PNG, save metadata
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
    num_image = 10000

    # Download images for each class
    download_images(class_name="myriapod",   taxon_id=144128, num_images=num_image)
    download_images(class_name="crustacean", taxon_id=85493, num_images=num_image)
    download_images(class_name="insect",     taxon_id=47158, num_images=num_image)
    download_images(class_name="arachnid",   taxon_id=47119, num_images=num_image)



