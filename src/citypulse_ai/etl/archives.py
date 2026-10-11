"""Official archive discovery, bounded downloads, and safe resumable extraction."""

from datetime import date
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter

OFFICIAL_BUCKET = "https://s3.amazonaws.com/tripdata/"
BLOCK_BYTES = 1024 * 1024


def fingerprint(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: dict) -> None:
    """Replace metadata atomically; retain the previous manifest on interruption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def month_range(start: str, end: str) -> list[str]:
    """Accept full-month local date boundaries, with an exclusive end."""
    left, right = date.fromisoformat(start), date.fromisoformat(end)
    if left.day != 1 or right.day != 1 or left >= right:
        raise ValueError("Use first-of-month dates with start < exclusive end")
    result = []
    while left < right:
        result.append(left.strftime("%Y-%m"))
        left = date(left.year + (left.month == 12), left.month % 12 + 1, 1)
    return result


def validate_month(month: str) -> str:
    parsed = date.fromisoformat(month + "-01")
    if parsed.strftime("%Y-%m") != month:
        raise ValueError("Month must be YYYY-MM")
    return month


def discover_archive(month: str) -> dict:
    """Use the official S3 listing rather than guessing a filename/schema."""
    prefix = validate_month(month).replace("-", "")
    url = OFFICIAL_BUCKET + "?list-type=2&prefix=" + prefix
    with urllib.request.urlopen(url, timeout=60) as response:
        root = ET.fromstring(response.read())
    candidates = []
    for item in root.findall("{*}Contents"):
        key = item.find("{*}Key").text
        if key == f"{prefix}-citibike-tripdata.zip":
            candidates.append({"month": month, "key": key,
                               "url": OFFICIAL_BUCKET + urllib.parse.quote(key),
                               "bytes": int(item.find("{*}Size").text),
                               "etag": item.find("{*}ETag").text,
                               "last_modified": item.find("{*}LastModified").text})
    if len(candidates) != 1:
        raise ValueError(f"Expected one modern NYC archive for {month}; found {len(candidates)}")
    return candidates[0]


class RemoteArchive(io.RawIOBase):
    """Seek a ZIP directory through small HTTP ranges, never downloading by accident."""

    def __init__(self, source: dict):
        self.source, self.position = source, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = (offset if whence == 0 else self.position + offset
                         if whence == 1 else self.source["bytes"] + offset)
        return self.position

    def read(self, size=-1):
        remaining = self.source["bytes"] - self.position
        size = min(remaining, remaining if size < 0 else size)
        if size <= 0:
            return b""
        request = urllib.request.Request(self.source["url"], headers={
            "Range": f"bytes={self.position}-{self.position + size - 1}",
            "If-Match": self.source["etag"],
        })
        with urllib.request.urlopen(request, timeout=60) as response:
            if response.status != 206:
                raise ValueError("Server ignored range request; refusing an unplanned full download")
            content_range = response.headers.get("Content-Range", "")
            if content_range != f"bytes {self.position}-{self.position + size - 1}/{self.source['bytes']}":
                raise ValueError("Unexpected range response")
            data = response.read(size)
        if len(data) != size:
            raise OSError("Truncated ZIP range")
        self.position += len(data)
        return data


def csv_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members = []
    seen = set()
    for member in archive.infolist():
        name = PurePosixPath(member.filename)
        if member.is_dir() or "__MACOSX" in name.parts or name.name.startswith("."):
            continue
        if name.suffix.lower() != ".csv":
            continue
        if name.is_absolute() or ".." in name.parts or "\\" in member.filename:
            raise ValueError("Unsafe ZIP member path")
        if member.filename in seen:
            raise ValueError("Duplicate ZIP member name")
        if (member.external_attr >> 16) & 0o170000 == 0o120000:
            raise ValueError("ZIP symlinks are not permitted")
        seen.add(member.filename)
        members.append(member)
    if not members:
        raise ValueError("Archive has no supported CSV files; inspect its structure manually")
    return sorted(members, key=lambda member: member.filename)


def inspect_archive(source: dict) -> dict:
    with zipfile.ZipFile(RemoteArchive(source)) as archive:
        members = csv_members(archive)
        return {"csv_bytes": sum(member.file_size for member in members),
                "members": [{"name": member.filename, "bytes": member.file_size,
                             "compressed_bytes": member.compress_size,
                             "crc32": f"{member.CRC:08x}"} for member in members]}


def probe_boundary_archive(month: str, prefix_bytes: int = 5 * 1024**2) -> dict:
    """Inspect bounded CSV prefixes; absence in a sample never establishes completeness."""
    from citypulse_ai.etl.schema import RAW_COLUMNS
    from citypulse_ai.etl.trips import _parse_timestamp
    source = discover_archive(month)
    members = []
    with zipfile.ZipFile(RemoteArchive(source)) as archive:
        for member in csv_members(archive):
            with archive.open(member) as stream:
                prefix = stream.read(prefix_bytes)
            prefix = prefix[:prefix.rfind(b"\n") + 1]
            reader = csv.DictReader(io.StringIO(prefix.decode("utf-8-sig")))
            if not set(RAW_COLUMNS).issubset(reader.fieldnames or []):
                raise ValueError("Boundary sample has unsupported schema")
            counts, examples, rows, invalid = Counter(), [], 0, 0
            for row in reader:
                rows += 1
                instant, error = _parse_timestamp(row["started_at"].strip())
                if error:
                    invalid += 1
                    continue
                start_month = instant.tz_convert("America/New_York").strftime("%Y-%m")
                counts[start_month] += 1
                if start_month != month and len(examples) < 5:
                    examples.append({key: row[key] for key in RAW_COLUMNS})
            members.append({"member": member.filename, "sample_rows": rows,
                            "prefix_bytes": len(prefix), "start_month_counts": dict(counts),
                            "invalid_start_timestamps": invalid, "boundary_examples": examples,
                            "columns": reader.fieldnames})
    return {"source": source, "members": members,
            "method": "Bounded leading CSV prefixes via HTTP ranges; parsed starts use unchanged Phase 1 policy",
            "limitations": "Not a full archive count or validation run; ZIP CRC cannot be verified from prefixes."}


def check_disk(path: Path, required_bytes: int, reserve_bytes: int) -> dict:
    parent = path.resolve()
    while not parent.exists():
        parent = parent.parent
    free = shutil.disk_usage(parent).free
    if free < required_bytes + reserve_bytes:
        raise OSError(f"Insufficient disk: {free:,} bytes free; need {required_bytes:,} + {reserve_bytes:,} reserve")
    return {"free_bytes": free, "estimated_additional_bytes": required_bytes,
            "reserve_bytes": reserve_bytes}


def download_archive(source: dict, directory: Path) -> tuple[Path, str]:
    """Resume with Range + If-Match. ETag is an identity token, not an MD5 checksum."""
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / source["key"]
    receipt = directory / (source["key"] + ".json")
    if target.exists():
        metadata = json.loads(receipt.read_text()) if receipt.exists() else {}
        if metadata.get("etag") != source["etag"] or metadata.get("bytes") != source["bytes"]:
            raise ValueError("Existing archive identity differs; preserve it and use a new raw directory")
        digest = fingerprint(target)
        if digest != metadata.get("sha256"):
            raise ValueError("Existing archive SHA-256 mismatch")
        return target, digest
    partial = directory / (source["key"] + ".part")
    partial_metadata = directory / (source["key"] + ".part.json")
    identity = {key: source[key] for key in ("url", "etag", "bytes")}
    if partial.exists():
        if not partial_metadata.exists() or json.loads(partial_metadata.read_text()) != identity:
            raise ValueError("Partial archive identity changed; keep it and use a new raw directory")
    else:
        write_json(partial_metadata, identity)
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > source["bytes"]:
        raise ValueError("Partial archive is larger than the official object")
    if offset < source["bytes"]:
        headers = {"If-Match": source["etag"]}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(source["url"], headers=headers)
        with urllib.request.urlopen(request, timeout=60) as response:
            if response.headers.get("ETag") != source["etag"]:
                raise ValueError("Source ETag changed during download")
            if offset and (response.status != 206 or response.headers.get("Content-Range") !=
                           f"bytes {offset}-{source['bytes'] - 1}/{source['bytes']}"):
                raise ValueError("Server did not honor resumable range; partial file left intact")
            if not offset and response.status != 200:
                raise ValueError("Unexpected archive response")
            with partial.open("ab") as stream:
                shutil.copyfileobj(response, stream, length=BLOCK_BYTES)
    if partial.stat().st_size != source["bytes"]:
        raise OSError("Incomplete archive download; rerun to resume")
    # SHA-256 provides local reproducibility; every CSV's ZIP CRC is checked on extraction.
    digest = fingerprint(partial)
    write_json(receipt, {**identity, "sha256": digest})
    partial.replace(target)
    return target, digest


def extract_archive(path: Path, destination: Path) -> list[dict]:
    """Retain raw CSVs; publish each only after full streaming ZIP CRC verification."""
    destination.mkdir(parents=True, exist_ok=True)
    receipt = destination / "extraction.json"
    archive_hash = fingerprint(path)
    previous = json.loads(receipt.read_text()) if receipt.exists() else {}
    if previous and previous.get("archive_sha256") != archive_hash:
        raise ValueError("Raw extraction belongs to a different archive")
    completed = {item["member"]: item for item in previous.get("files", [])}
    with zipfile.ZipFile(path) as archive:
        for member in csv_members(archive):
            target = destination.joinpath(*PurePosixPath(member.filename).parts)
            if target.exists():
                record = completed.get(member.filename)
                if record and fingerprint(target) == record["sha256"]:
                    continue
                raise ValueError(f"Unverified existing CSV: {target}; preserve it and use a new raw directory")
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".part")
            with archive.open(member) as source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output, length=BLOCK_BYTES)
            if temporary.stat().st_size != member.file_size:
                raise OSError("Truncated extraction")
            record = {"member": member.filename, "path": str(target.resolve()),
                      "bytes": member.file_size, "sha256": fingerprint(temporary),
                      "crc32": f"{member.CRC:08x}"}
            # Write receipt first so a crash immediately after rename can be resumed.
            completed[member.filename] = record
            write_json(receipt, {"archive_sha256": archive_hash, "files": list(completed.values())})
            temporary.replace(target)
    return [completed[name] for name in sorted(completed)]
