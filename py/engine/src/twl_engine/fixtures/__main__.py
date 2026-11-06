import argparse
from collections.abc import Sequence
from pathlib import Path

from twl_engine.fixtures.pack import write_pack
from twl_engine.fixtures.toy import build_toy_city
from twl_engine.goldens import demo_inputs
from twl_engine.run import run_seed_bytes
from twl_tape import tape_hash


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m twl_engine.fixtures",
        description="Build the toy city pack and print its pack_hash; with --tape, also run the "
        "demo launch on that city and print the tape hash.",
    )
    parser.add_argument("--seed", type=int, default=49)
    parser.add_argument("--residents", type=int, default=48_000)
    parser.add_argument("--stores", type=int, default=14)
    parser.add_argument("--out", type=Path, required=True, help="pack directory to write")
    parser.add_argument("--tape", type=Path, help="tape file to write for the demo launch")
    parser.add_argument("--tape-seed", type=int, default=1, help="run seed of the demo tape")
    args = parser.parse_args(argv)
    city = build_toy_city(seed=args.seed, residents=args.residents, stores=args.stores)
    print(write_pack(city, args.out))
    if args.tape is not None:
        data = run_seed_bytes(demo_inputs(args.tape_seed, city))
        args.tape.parent.mkdir(parents=True, exist_ok=True)
        args.tape.write_bytes(data)
        print(tape_hash(data))


if __name__ == "__main__":
    main()
