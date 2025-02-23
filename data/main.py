# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.



import requests
import pandas as pd
import random


def met(limit=5000):
    base_url = "https://collectionapi.metmuseum.org/public/collection/v1"

    # Get all available object IDs
    response = requests.get(f"{base_url}/objects").json()
    all_object_ids = response.get("objectIDs", [])

    if not all_object_ids:
        print("No objects found in the Met dataset.")
        return

    # Shuffle the dataset to get a random sample
    random.shuffle(all_object_ids)
    selected_object_ids = all_object_ids[:limit]

    artworks = []

    for count, object_id in enumerate(selected_object_ids, 1):
        obj_url = f"{base_url}/objects/{object_id}"
        obj_data = requests.get(obj_url).json()

        # Append only if the object is valid (API returns 'objectID' if valid)
        if "objectID" in obj_data:
            artworks.append({
                "objectID": obj_data.get("objectID", "NA") if obj_data.get("objectID") else "NA",
                "title": obj_data.get("title", "NA") if obj_data.get("title") else "NA",
                "artist": obj_data.get("artistDisplayName", "NA") if obj_data.get("artistDisplayName") else "NA",
                "medium": obj_data.get("medium", "NA") if obj_data.get("medium") else "NA",
                "department": obj_data.get("department", "NA") if obj_data.get("department") else "NA",
                "culture": obj_data.get("culture", "NA") if obj_data.get("culture") else "NA",
                "period": obj_data.get("period", "NA") if obj_data.get("period") else "NA",
                "classification": obj_data.get("classification", "NA") if obj_data.get("classification") else "NA",
                "imageURL": obj_data.get("primaryImage", "NA") if obj_data.get("primaryImage") else "NA"
            })
        # Progress tracking
        if count % 100 == 0:
            print(f"Fetched {count} artworks so far...")

    # Convert to DataFrame and save
    df = pd.DataFrame(artworks)
    df.to_csv("met_museum_data.csv", index=False)
    print(f"Met Museum data saved, Total objects: {len(df)}")

def chicago(limit=5000):
    base_url = "https://api.artic.edu/api/v1/artworks"
    per_page = 100  # Number of artworks per API request
    artworks = []
    page = 1

    while len(artworks) < limit:
        response = requests.get(f"{base_url}?page={page}&limit={per_page}").json()

        if "data" not in response or not response["data"]:  # Stop if no more data
            break

        for artwork in response["data"]:
            if len(artworks) >= limit:  # Stop if we reach the limit
                break

            artworks.append({
                "objectID": artwork.get("id", "NA"),
                "title": artwork.get("title", "NA"),
                "artist": artwork["artist_title"] if artwork.get("artist_title") else "NA",
                "medium": artwork.get("medium_display", "NA"),
                "department": artwork.get("department_title", "NA"),
                "culture": artwork.get("place_of_origin", "NA"),
                "period": artwork.get("date_display", "NA"),
                "classification": artwork.get("classification_title", "NA"),
                "imageURL": f"https://www.artic.edu/iiif/2/{artwork['image_id']}/full/843,/0/default.jpg"
                if artwork.get("image_id") else "NA"
            })

        print(f"Fetched {len(artworks)} artworks so far...")
        page += 1  # Go to the next page

    df = pd.DataFrame(artworks)
    df.to_csv("art_institute_chicago_data.csv", index=False)
    print(f"Art Institute of Chicago data saved! Total records: {len(df)}")


def cleveland(limit=5000):
    base_url = "https://openaccess-api.clevelandart.org/api/artworks/"

    artworks = []
    page = 1
    page_size = 100  # Number of items per request

    print("Fetching artworks...")

    while len(artworks) <= limit:
        print(f"Fetching page {page}... (Total collected: {len(artworks)})")

        # Fetch data
        response = requests.get(f"{base_url}?page={page}&limit={page_size}").json()

        # Stop if no more data
        if "data" not in response or not response["data"]:
            break

            # Process artworks on this page
        for artwork in response["data"]:
            object_id = artwork.get("id", "NA")
            title = artwork.get("title", "NA")

            # Extract artist safely
            artist = "NA"
            if "creators" in artwork and isinstance(artwork["creators"], list) and artwork["creators"]:
                artist = artwork["creators"][0].get("description", "NA")

            # Extract other metadata
            medium = artwork.get("technique", "NA")
            department = artwork.get("department", "NA")
            culture = artwork.get("culture", "NA")
            classification = artwork.get("type", "NA")

            # Extract image URL safely
            image_url = "NA"
            if "images" in artwork and isinstance(artwork["images"], list) and artwork["images"]:
                image_url = artwork["images"][0].get("url", "NA")

            # Append artwork data
            artworks.append({
                "objectID": object_id,
                "title": title,
                "artist": artist,
                "medium": medium,
                "department": department,
                "culture": culture,
                "period": "NA",  # Not provided in API
                "classification": classification,
                "imageURL": image_url
            })

        page += 1


    random.shuffle(artworks)

    selected_artworks = artworks[:limit]

    # Convert to DataFrame & Save to CSV
    df = pd.DataFrame(selected_artworks)
    df.to_csv("cleveland_museum_data.csv", index=False)

    print(f" Saved {len(df)} randomly selected artworks to cleveland_museum_data.csv!")

def victoria_albert(limit=5000):
    search_url = "https://api.vam.ac.uk/v2/objects/search?page_size=100&q=*"
    object_ids = []
    page = 1

    # Step 1: Collect all available object IDs
    while len(object_ids) < limit * 2:  # Collect extra IDs for randomness
        response = requests.get(f"{search_url}&page={page}").json()
        if "records" not in response or not response["records"]:
            break  # Stop if no more data

        object_ids.extend([obj["systemNumber"] for obj in response["records"]])
        print(f"Collected {len(object_ids)} object IDs so far...")
        page += 1


    random.shuffle(object_ids)
    selected_ids = object_ids[:limit]

    artworks = []
    for obj_id in selected_ids:
        obj_url = f"https://api.vam.ac.uk/v2/object/{obj_id}"
        obj_response = requests.get(obj_url).json()
        if "record" not in obj_response:
            continue  # Skip missing data
        obj = obj_response["record"]
        title = obj.get("briefDescription", "NA") or obj.get("_primaryTitle", "NA")
        artist = obj["artistMakerPerson"][0]["name"]["text"] if obj.get("artistMakerPerson") else "NA"
        medium = obj.get("materialsAndTechniques", "NA")
        department = obj.get("collectionCode", {}).get("text", "NA")
        culture = obj["placesOfOrigin"][0]["place"]["text"] if obj.get("placesOfOrigin") else "NA"
        period = obj["productionDates"][0]["date"]["text"] if obj.get("productionDates") else "NA"
        classification = obj.get("objectType", "NA")
        # Extracting image URL
        imageURL = obj.get("meta", {}).get("images", {}).get("_primary_thumbnail", "NA")
        artworks.append({
            "objectID": obj.get("systemNumber", "NA"),
            "title": title,
            "artist": artist,
            "medium": medium,
            "department": department,
            "culture": culture,
            "period": period,
            "classification": classification,
            "imageURL": imageURL
        })

        #print(f"Fetched {len(artworks)} artworks so far...")
    df = pd.DataFrame(artworks)
    df.to_csv("victoria_albert_museum_data.csv", index=False)
    print(f"Victoria & Albert Museum data saved! Total records: {len(df)}")

def combined():
    chicago_df = pd.read_csv("art_institute_chicago_data.csv")
    cleveland_df = pd.read_csv("cleveland_museum_data.csv")
    met_df = pd.read_csv("met_museum_data.csv")
    va_df = pd.read_csv("victoria_albert_museum_data.csv")
    combined_df = pd.concat([chicago_df, cleveland_df, met_df, va_df], ignore_index=True)
    combined_df.to_csv("merged_museum_data.csv", index=False)

if __name__ == '__main__':
    print_hi('PyCharm')
    combined()

