import hashlib
import shutil
import urllib.request
from pathlib import Path


def file_md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url: str, md5: str, dest: Path) -> Path:
    """Download `url` to `dest` once; a cached file must still match the pinned checksum."""
    if dest.exists():
        if file_md5(dest) != md5:
            raise ValueError(f"{dest} does not match pinned md5 {md5}; delete it to refetch")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as out:
        shutil.copyfileobj(response, out, 1 << 20)
    if (actual := file_md5(partial)) != md5:
        partial.unlink()
        raise ValueError(f"{url} has md5 {actual}, pinned {md5}")
    partial.rename(dest)
    return dest
