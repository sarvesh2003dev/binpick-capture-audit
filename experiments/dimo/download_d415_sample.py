"""Fetch the official DIMO D415 sample with bounded HTTP ranges, not a full tar.

python download_d415_sample.py --data data --local-prefix ../../work/dimo_data/dimo_sample.tar.partial
The optional local prefix must be the initial bytes of the same uncompressed tar.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

URL = "https://datasets.ilabt.imec.be/dimo/tar/dimo_sample.tar"
PREFIX = "dimo_sample/real_d415/000042/"
LIMIT = 180_000_000


class RangeReader(io.RawIOBase):
    """Seekable, bounded, uncompressed HTTP archive with optional prefix cache."""
    def __init__(self, url, local_prefix=None):
        self.url = url
        self.session = requests.Session()
        self.session.mount("https://", HTTPAdapter(max_retries=Retry(
            total=2, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])))
        self.pos = self.transferred = self.requests = 0
        self.size = None
        self.cache_start, self.cache = 0, b""
        self.local = None
        self.local_size = 0
        self.etag = None
        probe = self.fetch(0, 1023)
        if local_prefix and local_prefix.exists():
            self.local = local_prefix.open("rb")
            self.local_size = local_prefix.stat().st_size
            if self.local_size > self.size or self.local.read(len(probe)) != probe:
                raise RuntimeError("Local prefix does not match the official archive")
        self.cache = probe

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        target = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        if target < 0 or target > self.size:
            raise ValueError("Seek outside archive")
        self.pos = target
        return target

    def fetch(self, start, end):
        if end - start + 1 + self.transferred > LIMIT or self.requests >= 512:
            raise RuntimeError("Bounded download limit exceeded")
        headers = {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}
        if self.etag:
            headers["If-Range"] = self.etag
        with self.session.get(self.url, headers=headers, timeout=(20, 60), stream=True) as response:
            if response.status_code != 206:
                raise RuntimeError(f"Server ignored range: HTTP {response.status_code}")
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
            if not match:
                raise RuntimeError("Invalid Content-Range")
            first, last, total = map(int, match.groups())
            if first != start or last != end or (self.size is not None and total != self.size):
                raise RuntimeError("Unexpected range or archive changed")
            if total > 500_000_000:
                raise RuntimeError("Unexpected archive size")
            self.size = total
            self.etag = self.etag or response.headers.get("ETag")
            body = response.raw.read(end - start + 2)
            if len(body) != end - start + 1:
                raise RuntimeError("Incomplete or oversized range response")
        self.transferred += len(body)
        self.requests += 1
        return body

    def read(self, size=-1):
        if size < 0:
            raise ValueError("Unbounded reads forbidden")
        size = min(size, self.size - self.pos)
        if not size:
            return b""
        if self.local and self.pos + size <= self.local_size:
            self.local.seek(self.pos)
            body = self.local.read(size)
        elif self.cache_start <= self.pos and self.pos + size <= self.cache_start + len(self.cache):
            start = self.pos - self.cache_start
            body = self.cache[start:start + size]
        else:
            self.cache_start = self.pos
            self.cache = self.fetch(self.pos, min(self.size - 1, self.pos + max(size, 262144) - 1))
            body = self.cache[:size]
        if len(body) != size:
            raise IOError("Short archive read")
        self.pos += size
        return body

    def close(self):
        if self.local:
            self.local.close()
        self.session.close()
        super().close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--local-prefix", type=Path)
    args = parser.parse_args()
    destination = (args.data / "selected").resolve()
    destination.mkdir(parents=True, exist_ok=True)
    selected = []
    started = time.monotonic()
    with RangeReader(URL, args.local_prefix) as source:
        with tarfile.open(fileobj=source, mode="r:") as archive:
            for member in archive:
                if not member.name.startswith(PREFIX) or not member.isfile():
                    continue
                relative = PurePosixPath(member.name)
                if relative.name.startswith("."):
                    continue
                if relative.is_absolute() or ".." in relative.parts or "\\" in member.name:
                    raise RuntimeError("Unsafe archive member")
                leaf = relative.relative_to(PurePosixPath(PREFIX))
                is_image = len(leaf.parts) == 2 and leaf.parts[0] in {"rgb", "depth"} and leaf.suffix == ".png"
                is_label = len(leaf.parts) == 1 and leaf.suffix == ".json"
                if not (is_image or is_label) or member.size > 20_000_000:
                    raise RuntimeError(f"Unexpected selected file: {member.name}")
                if sum(item["bytes"] for item in selected) + member.size > LIMIT:
                    raise RuntimeError("Selected size limit exceeded")
                target = destination.joinpath(*relative.parts).resolve()
                if not target.is_relative_to(destination):
                    raise RuntimeError("Path escaped output directory")
                with archive.extractfile(member) as content:
                    data = content.read()
                if len(data) != member.size:
                    raise RuntimeError("Incomplete selected file")
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(target.suffix + ".partial")
                temporary.write_bytes(data)
                temporary.replace(target)
                selected.append({"name": member.name, "bytes": member.size,
                                 "sha256": hashlib.sha256(data).hexdigest()})
                print(f"{len(selected):02d}: {leaf} | network {source.transferred / 1e6:.1f} MB", flush=True)
                if len(selected) == 30:
                    break
        expected_ids = {f"{i:06d}.png" for i in range(13)}
        for kind in ("rgb", "depth"):
            actual = {PurePosixPath(item["name"]).name for item in selected if f"/{kind}/" in item["name"]}
            if actual != expected_ids:
                raise RuntimeError(f"Expected exactly 13 paired {kind} views")
        labels = {PurePosixPath(item["name"]).name for item in selected if item["name"].endswith(".json")}
        if labels != {"scene_gt_world.json", "scene_gt.json", "scene_info.json", "scene_camera.json"}:
            raise RuntimeError("Unexpected label set")
        provenance = {
            "dataset": "Dataset of Industrial Metal Objects (DIMO)",
            "authors": ["Peter De Roovere", "Steven Moonen", "Nick Michiels", "Francis Wyffels"],
            "project": "https://pderoovere.github.io/dimo/",
            "license": "CC BY 4.0", "license_source": "https://pderoovere.github.io/dimo/usage/",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "paper": "https://arxiv.org/abs/2208.04052",
            "source_url": URL, "selection": PREFIX, "members": selected,
            "archive_bytes": source.size, "network_bytes": source.transferred,
            "range_requests": source.requests, "archive_etag": source.etag,
            "local_prefix_bytes_available": source.local_size,
            "selection_modified": False,
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "note": "Hashes authenticate local selected bytes; no publisher checksums were provided. RGB/depth and labels are original unmodified dataset files.",
        }
    (args.data / "d415_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"Ready: 13 D415 RGB/depth pairs and 4 labels; {provenance['network_bytes'] / 1e6:.1f} MB fetched", flush=True)


if __name__ == "__main__":
    main()
