"""
Script to download images from INaturalist
"""

from PIL import Image
import requests
import os
import json

# Function to download bee images and save them in the appropriate folder structure
def download_images(folderName, taxon_id, per_page=50, page=1):
    # Define folder paths
    jpg_folder      = f'images/{folderName}/jpg'
    png_folder      = f'images/{folderName}/png'
    metadata_folder = f'images/{folderName}/metadata'

    # Create directories if they don't exist
    os.makedirs(jpg_folder, exist_ok=True)
    os.makedirs(png_folder, exist_ok=True)
    os.makedirs(metadata_folder, exist_ok=True)

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
            for photo in result['photos']:
                # Get the image URL and download the JPEG
                img_url       = photo['url'].replace('square', 'original')
                img_data      = requests.get(img_url).content
                img_file_name = f'{folderName}_{i + (page - 1) * per_page}'
                
                # Save JPEG image
                jpg_file_path = os.path.join(jpg_folder, f'{img_file_name}.jpg')
                with open(jpg_file_path, 'wb') as img_file:
                    img_file.write(img_data)

                print(f"Downloaded JPEG: {jpg_file_path}")

                # Convert JPEG to PNG
                img           = Image.open(jpg_file_path)
                png_file_path = os.path.join(png_folder, f'{img_file_name}.png')
                img.save(png_file_path, 'PNG')

                print(f"Converted to PNG: {png_file_path}")

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
    numImages = 200

    # Download images for each class
    download_images(folderName="bee",    taxon_id=630955, per_page=numImages);
    download_images(folderName="spider", taxon_id=47118,  per_page=numImages);
    download_images(folderName="wasp",   taxon_id=52747,  per_page=numImages);
    download_images(folderName="beetle", taxon_id=47208,  per_page=numImages);
    download_images(folderName="moth",   taxon_id=47157,  per_page=numImages);



