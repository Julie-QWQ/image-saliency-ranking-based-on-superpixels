import argparse
import os
import sys
import urllib.request
import zipfile

DATASETS = {
    "MSRA-B": "http://mftp.mmcheng.net/Data/MSRA-B.zip",
    "DUTS-TE": "http://saliencydetection.net/duts/download/DUTS-TE.zip",
}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def download(url, out_path):
    ensure_dir(os.path.dirname(out_path))
    print(f"Downloading {url} -> {out_path}")
    urllib.request.urlretrieve(url, out_path)


def extract(zip_path, out_dir):
    print(f"Extracting {zip_path} -> {out_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/raw")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--datasets", nargs="*", default=["MSRA-B", "DUTS-TE"])
    args = parser.parse_args()

    for name in args.datasets:
        if name not in DATASETS:
            print(f"Unknown dataset: {name}")
            sys.exit(1)
        url = DATASETS[name]
        zip_path = os.path.join(args.root, f"{name}.zip")
        if not os.path.exists(zip_path):
            download(url, zip_path)
        else:
            print(f"Skip existing: {zip_path}")
        if args.extract:
            out_dir = os.path.join(args.root, name)
            if not os.path.exists(out_dir):
                extract(zip_path, out_dir)
            else:
                print(f"Skip existing folder: {out_dir}")


if __name__ == "__main__":
    main()
