import requests
import pandas as pd

# Replace with your API Key
PEXELS_API_KEY = "CddO0QGAXFNgdsSKjIgephbMBZ2zgSLpATAAHTLYtuUl7tCAgIAngXg3"

# Define the search query
jkl =['cat','dog']


# Create an empty list to store image data
image_data = []
for k in jkl:
    query = k
    # Loop through pages (Pexels API allows max 80 images per page)
    for i in range(1, 20):  # Adjust the range as needed (avoid excessive requests)

        url = f"https://api.pexels.com/v1/search?query={query}&per_page=20&page={i}"  # Fetch 10 images per page

        # Set headers
        headers = {"Authorization": PEXELS_API_KEY}

        # Make the API request
        response = requests.get(url, headers=headers)

        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()
            
            # Extract and store image metadata
            for photo in data["photos"]:
                image_data.append({
                    "id": photo["id"],
                    "photographer": photo["photographer"],
                    "photographer_url": photo["photographer_url"],
                    "image_url": photo["src"]["original"]
                })
                print()
        else:
            print("Error:", response.status_code, response.text)

# Convert list to DataFrame
df = pd.DataFrame(image_data)

# Display DataFrame
print(df.head())


df.to_excel("pexels_images_combined.xlsx", index=False)
