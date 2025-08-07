import argparse
import requests
import os
import tarfile

def download_file_from_google_drive(file_id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)
    print(f"Downloaded to '{destination}'")

def extract_tar_gz(archive_path, extract_path):
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=extract_path)
    print(f"Tar extracted to : {extract_path} ")

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination, chunk_size=32768):
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                f.write(chunk)

def parse_args():
    parser = argparse.ArgumentParser(description="Download biosyn data")
    parser.add_argument("--ds_name", 
                        required=True, 
                        choices=["ncbi-disease", "bc5dr-disease", "bc5dr-chemical"], 
                        help="name of dataset",
                        default="ncbi-disease"
                        )
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    files  = {
        "ncbi-disease": "1mmV7p33E1iF32RzAET3MsLHPz1PiF9vc",
        "bc5dr-disease": "1moAqukbrdpAPseJc3UELEY6NLcNk22AA",
        "bc5dr-chemical": "1mgQhjAjpqWLCkoxIreLnNBYcvjdsSoGi"
    }
    args = parse_args()
    file_id = files[args.ds_name]
    dir = f"./data/{args.ds_name}"
    destination = f"{dir}/downloaded_file.tar.gz"
    os.makedirs(dir ,exist_ok=True)
    download_file_from_google_drive(file_id, destination)
    extract_tar_gz(destination, "./data")


# python download_ds.py --ds_name ncbi-disease