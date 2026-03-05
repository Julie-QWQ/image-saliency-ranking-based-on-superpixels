import argparse
import os
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile

DATASETS = {
    "MSRA-B": {
        "urls": [
            "http://mftp.mmcheng.net/Data/MSRA-B.zip",
            "https://mftp.mmcheng.net/Data/MSRA-B.zip",
        ],
        "referer": "http://mftp.mmcheng.net/",
    },
    "DUTS-TE": {
        "urls": [
            # Official DUTS mirror
            "http://saliencydetection.net/duts/download/DUTS-TE.zip",
            "https://saliencydetection.net/duts/download/DUTS-TE.zip",
        ],
        "referer": "http://saliencydetection.net/",
    },
}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def download(url, out_path, referer=None):
    ensure_dir(os.path.dirname(out_path))
    print(f"Downloading {url} -> {out_path}")
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
            **({"Referer": referer} if referer else {}),
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp, open(out_path, "wb") as f:
            f.write(resp.read())
        return
    except urllib.error.HTTPError as exc:
        print(f"HTTPError {exc.code}: {exc.reason}, fallback to curl/wget")
    except urllib.error.URLError as exc:
        print(f"URLError: {exc.reason}, fallback to curl/wget")

    _download_with_cli(url, out_path, referer)


def _download_with_cli(url, out_path, referer=None):
    if _has_cmd("curl"):
        cmd = ["curl", "-L", "-A", "Mozilla/5.0", "-f"]
        if referer:
            cmd.extend(["-e", referer])
        cmd.extend([url, "-o", out_path])
        _run(cmd)
        return
    if _has_cmd("wget"):
        cmd = ["wget", "-O", out_path, "--user-agent=Mozilla/5.0"]
        if referer:
            cmd.append(f"--referer={referer}")
        cmd.append(url)
        _run(cmd)
        return
    raise RuntimeError("Neither curl nor wget is available for download fallback.")


def _has_cmd(name):
    return subprocess.call(["which", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0


def _run(cmd):
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)


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
        dataset_cfg = DATASETS[name]
        zip_path = os.path.join(args.root, f"{name}.zip")
        if not os.path.exists(zip_path):
            _download_with_fallbacks(name, dataset_cfg, zip_path)
        else:
            print(f"Skip existing: {zip_path}")
        if args.extract:
            out_dir = os.path.join(args.root, name)
            if not os.path.exists(out_dir):
                extract(zip_path, out_dir)
            else:
                print(f"Skip existing folder: {out_dir}")


def _download_with_fallbacks(name, dataset_cfg, zip_path):
    for url in dataset_cfg["urls"]:
        try:
            download(url, zip_path, dataset_cfg.get("referer"))
        except Exception as exc:
            print(f"Download failed for {url}: {exc}")
            _cleanup_bad_file(zip_path)
            continue
        if zipfile.is_zipfile(zip_path):
            return
        print(f"Invalid zip file for {name}, trying next mirror...")
        _cleanup_bad_file(zip_path)
    raise RuntimeError(f"All mirrors failed for {name}")


def _cleanup_bad_file(path):
    if os.path.exists(path):
        os.remove(path)


if __name__ == "__main__":
    main()
