import requests
import os

# Function to download research-grade spider images using iNaturalist API
def download_spider_images(taxon_id=47118, per_page=50, page=1):
    # Create a directory to save images
    if not os.path.exists('spider_images'):
        os.makedirs('spider_images')

    # iNaturalist API endpoint with research grade filtering
    url = f'https://api.inaturalist.org/v1/observations?taxon_id={taxon_id}&quality_grade=research&per_page={per_page}&page={page}'

    # Make the request to the API
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        return

    # Parse the JSON response
    data = response.json()

    # Download images from the API response
    for i, result in enumerate(data['results']):
        if 'photos' in result:
            for photo in result['photos']:
                img_url = photo['url'].replace('square', 'original')
                img_data = requests.get(img_url).content
                img_file_path = f'spider_images/spider_{i + (page - 1) * per_page}.jpg'

                # Save the image
                with open(img_file_path, 'wb') as img_file:
                    img_file.write(img_data)

                print(f"Downloaded: {img_file_path}")

if __name__ == "__main__":
    download_spider_images()
