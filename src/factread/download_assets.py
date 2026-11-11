"""Download and verify the fixed inputs and trained parameters."""
import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path
from urllib.request import urlopen
from urllib.parse import urlsplit

DEFAULT_SOURCE = None


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(Path(__file__).resolve().parents[2] / "artifacts/manifest.json"))
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--repository-url", help="Anonymous repository URL, when archive parts are not present locally")
    parser.add_argument("--output", default="assets")
    parser.add_argument("--cache", default=".asset_download")
    args = parser.parse_args()
    if args.repository_url:
        url = urlsplit(args.repository_url)
        elements = url.path.strip("/").split("/")
        if url.scheme != "https" or url.netloc != "anonymous.4open.science" or len(elements) != 2 or elements[0] != "r":
            parser.error("Expected an https://anonymous.4open.science/r/<repository-id>/ URL")
        args.source = f"https://{url.netloc}/api/repo/{elements[1]}/file/"
    manifest = json.loads(Path(args.manifest).read_text())
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use an empty output directory")
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    resolved_parts = []
    for part in manifest["parts"]:
        local = Path(args.manifest).resolve().parent / Path(part["path"]).name
        destination = local if local.exists() else cache / Path(part["path"]).name
        if not destination.exists() or digest(destination) != part["sha256"]:
            if not args.source:
                raise FileNotFoundError("Download the complete repository ZIP, including artifacts/, or supply --source for the asset endpoint")
            temporary = destination.with_suffix(".tmp")
            with urlopen(args.source.rstrip("/") + "/" + part["path"] + "?download=1", timeout=120) as response, temporary.open("wb") as stream:
                shutil.copyfileobj(response, stream, 8 * 1024 * 1024)
            if digest(temporary) != part["sha256"] or temporary.stat().st_size != part["bytes"]:
                temporary.unlink()
                raise ValueError("Asset part checksum mismatch")
            temporary.replace(destination)
        resolved_parts.append(destination)
        print("Verified " + destination.name, flush=True)
    archive = cache / "assets.tar.gz"
    with archive.open("wb") as stream:
        for path in resolved_parts:
            with path.open("rb") as source:
                shutil.copyfileobj(source, stream, 8 * 1024 * 1024)
    if digest(archive) != manifest["archive_sha256"]:
        raise ValueError("Asset archive checksum mismatch")
    output.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as bundle:
        for entry in bundle.getmembers():
            target = (output / entry.name).resolve()
            if not target.is_relative_to(output.resolve()) or not entry.isfile():
                raise ValueError("Invalid archive entry")
            with bundle.extractfile(entry) as source:
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("wb") as stream:
                    shutil.copyfileobj(source, stream, 8 * 1024 * 1024)
    files = json.loads((output / "manifest.json").read_text())["files"]
    for name, spec in files.items():
        path = output / name
        if digest(path) != spec["sha256"] or path.stat().st_size != spec["bytes"]:
            raise ValueError("Extracted asset checksum mismatch: " + name)
    print("All reproduction assets verified", flush=True)


if __name__ == "__main__":
    main()
