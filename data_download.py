import requests
import zipfile
import io
import os

# WikiText-2 (Raw version)
url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip"
save_dir = "wikitext-2"

if not os.path.exists(save_dir):
    print(f"Downloading WikiText-2...")
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(save_dir)
    print("Unzipped!")
    
    # Read the train file to verify
    with open(f"{save_dir}/wikitext-2-raw/wiki.train.raw", 'r', encoding='utf-8') as f:
        data = f.read()
    print(f"Train data size: {len(data) / 1024 / 1024:.2f} MB")
    
    # Optional: Combine train/test/val into one big input.txt for your current code
    with open("wikitext_full.txt", "w", encoding="utf-8") as outfile:
        outfile.write(data)
    print("Saved combined file to 'wikitext_full.txt'")
else:
    print("WikiText-2 already downloaded.")