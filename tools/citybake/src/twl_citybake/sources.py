import hashlib
import shutil
import urllib.request
import zipfile
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
    try:
        with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as out:
            shutil.copyfileobj(response, out, 1 << 20)
        if (actual := file_md5(partial)) != md5:
            raise ValueError(f"{url} has md5 {actual}, pinned {md5}")
        partial.rename(dest)
    finally:
        partial.unlink(missing_ok=True)
    return dest


def extract_member(archive: Path, member: str, dest: Path) -> Path:
    """Unpack one named file of a zip to `dest`, once; `member` is looked up, never joined to a
    path, so the archive cannot write elsewhere."""
    if dest.exists():
        return dest
    partial = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(archive) as zf, zf.open(member) as src, partial.open("wb") as out:
            shutil.copyfileobj(src, out, 1 << 20)
        partial.rename(dest)
    except KeyError as missing:
        raise ValueError(f"{archive} has no member {member!r}") from missing
    finally:
        partial.unlink(missing_ok=True)
    return dest
