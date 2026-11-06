"""City pack v2 constants (ARCHITECTURE.md section 5)."""

PACK_FORMAT = "twinlight-pack"
PACK_VERSION = 2
COMPRESSION = "zstd"
ZSTD_LEVEL = 19
COLUMN_ALIGN = 8

DEFAULT_TILE_LEVEL = 6
MAX_TILE_LEVEL = 8

FIRST_LOAD_BUDGET = 6_000_000
ENGINE_BUDGET = 10_000_000

NO_PERSONA = 0xFFFFFFFF
FLOOR_HEIGHT_M = 3.2
MAX_DISTRICTS = 255

FIRST_LOAD_FILES = (
    "buildings.bin",
    "districts.json",
    "residents.bin",
    "river.bin",
    "roads.bin",
    "stores.bin",
)
ENGINE_FILES = ("graph.bin", "hexes.bin", "residents_engine.bin", "travel.bin")
DETAIL_FILE = "buildings_detail.bin"


class PackError(ValueError):
    """The input cannot be written as a pack, or the files are not an intact pack."""
