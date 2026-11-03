import argparse
from collections.abc import Sequence
from pathlib import Path

from twl_engine.fixtures.pack import write_pack
from twl_engine.fixtures.toy import build_toy_city


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m twl_engine.fixtures",
        description="Build the toy city pack and print its pack_hash.",
    )
    parser.add_argument("--seed", type=int, default=49)
    parser.add_argument("--residents", type=int, default=48_000)
    parser.add_argument("--stores", type=int, default=14)
    parser.add_argument("--out", type=Path, required=True, help="pack directory to write")
    args = parser.parse_args(argv)
    city = build_toy_city(seed=args.seed, residents=args.residents, stores=args.stores)
    print(write_pack(city, args.out))


if __name__ == "__main__":
    main()
