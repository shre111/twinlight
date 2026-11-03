import argparse
from collections.abc import Sequence
from importlib.metadata import version


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="citybake",
        description="Bake a Twinlight city pack from OpenStreetMap, population and persona data.",
    )
    parser.add_argument("--version", action="version", version=version("twl-citybake"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    build_parser().parse_args(argv)
    return 0
