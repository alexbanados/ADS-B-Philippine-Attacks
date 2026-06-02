from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from mod_tuner import ADJUSTABLE_ALPHA
from mod_tuner import ADJUSTABLE_ATTACK_DURATION
from mod_tuner import ADJUSTABLE_ATTACK_DURATION_MAX
from mod_tuner import ADJUSTABLE_ATTACK_DURATION_MIN
from mod_tuner import ADJUSTABLE_ATTACK_START_BIN
from mod_tuner import ADJUSTABLE_ATTACK_START_BIN_MAX
from mod_tuner import ADJUSTABLE_ATTACK_START_BIN_MIN
from mod_tuner import ADJUSTABLE_BIN_COUNT
from mod_tuner import ADJUSTABLE_DIRECTION
from mod_tuner import ADJUSTABLE_DURATION_AMPLITUDE_BETA
from mod_tuner import ADJUSTABLE_ENVELOPE_TYPE
from mod_tuner import ADJUSTABLE_K
from mod_tuner import ADJUSTABLE_RANDOM_SEED
from mod_tuner import ADJUSTABLE_SAMPLE_SIZE
from mod_tuner import ENVELOPE_CHOICES
from mod_tuner import generate_scalar_feature_attack
from mod_tuner import load_feature_std_lookup
from mod_tuner import prefixed_output_name
from mod_tuner import random_attack_direction
from mod_tuner import random_attack_duration
from mod_tuner import random_attack_start_bin
from mod_tuner import select_input_csvs
from mod_tuner import validate_common_attack_settings


ALTITUDE_FEATURE_NAME = "altitude_meters"
ATTACK_TYPE = "modified_altitude"
OUTPUT_FILENAME_PREFIX = "modalt_"
ADJUSTABLE_LABEL_MOD_ALT = 1


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate modified-altitude attack CSVs from a random sample of "
            "authentic flight CSVs."
        )
    )
    parser.add_argument(
        "input_folder",
        type=Path,
        help="Authentic one-flight CSV file or folder containing CSV files.",
    )
    parser.add_argument(
        "--stats",
        type=Path,
        required=True,
        help="Attack tuning statistics CSV to use for this run.",
    )
    parser.add_argument(
        "--output-folder",
        type=Path,
        required=True,
        help="Folder where modified-altitude CSVs will be written.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=ADJUSTABLE_BIN_COUNT,
        help=f"Number of route-progress bins. Default: {ADJUSTABLE_BIN_COUNT}.",
    )
    parser.add_argument(
        "--attack-start-bin",
        type=int,
        default=ADJUSTABLE_ATTACK_START_BIN,
        help=(
            "Optional fixed first route-progress bin in the attack window. "
            "Default: random integer from "
            f"{ADJUSTABLE_ATTACK_START_BIN_MIN} to "
            f"{ADJUSTABLE_ATTACK_START_BIN_MAX} for each CSV."
        ),
    )
    parser.add_argument(
        "--attack-duration",
        type=int,
        default=ADJUSTABLE_ATTACK_DURATION,
        help=(
            "Optional fixed number of bins in the attack window. "
            "Default: random integer from "
            f"{ADJUSTABLE_ATTACK_DURATION_MIN} to "
            f"{ADJUSTABLE_ATTACK_DURATION_MAX} for each CSV."
        ),
    )
    parser.add_argument(
        "--k",
        type=float,
        default=ADJUSTABLE_K,
        help=f"Attack intensity multiplier. Default: {ADJUSTABLE_K}.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=ADJUSTABLE_ALPHA,
        help=f"Gaussian randomness multiplier. Default: {ADJUSTABLE_ALPHA}.",
    )
    parser.add_argument(
        "--direction",
        type=int,
        choices=(-1, 1),
        default=ADJUSTABLE_DIRECTION,
        help=(
            "Optional fixed attack direction: +1 for altitude increase, -1 for "
            "altitude decrease. Default: random choice of -1 or +1 for each CSV."
        ),
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=ADJUSTABLE_SAMPLE_SIZE,
        help=f"Number of random CSVs to process. Default: {ADJUSTABLE_SAMPLE_SIZE}.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=ADJUSTABLE_RANDOM_SEED,
        help=(
            "Random seed for file sampling and Gaussian offsets. "
            f"Default: {ADJUSTABLE_RANDOM_SEED}."
        ),
    )
    parser.add_argument(
        "--label",
        type=int,
        default=ADJUSTABLE_LABEL_MOD_ALT,
        help=(
            "Class label for modified altitude rows. "
            f"Default: {ADJUSTABLE_LABEL_MOD_ALT}."
        ),
    )
    parser.add_argument(
        "--duration-amplitude-beta",
        type=float,
        default=ADJUSTABLE_DURATION_AMPLITUDE_BETA,
        help=(
            "Scale attack amplitude by effective_duration / 500 raised to this "
            "power. Larger values make short attacks quieter and long attacks "
            f"stronger. Default: {ADJUSTABLE_DURATION_AMPLITUDE_BETA}."
        ),
    )
    parser.add_argument(
        "--envelope-type",
        choices=ENVELOPE_CHOICES,
        default=ADJUSTABLE_ENVELOPE_TYPE,
        help=(
            "Attack envelope shape. random chooses one envelope per CSV. "
            f"Default: {ADJUSTABLE_ENVELOPE_TYPE}."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(sys.argv[1:] if argv is None else argv)
        validate_common_attack_settings(
            input_folder=args.input_folder,
            output_folder=args.output_folder,
            stats_csv=args.stats,
            bins=args.bins,
            attack_start_bin=args.attack_start_bin,
            attack_duration=args.attack_duration,
            k=args.k,
            alpha=args.alpha,
            sample_size=args.sample_size,
            duration_amplitude_beta=args.duration_amplitude_beta,
        )

        file_rng = np.random.default_rng(args.seed)
        attack_rng = np.random.default_rng(args.seed + 1)
        input_csvs = select_input_csvs(args.input_folder, args.sample_size, file_rng)
        std_lookup = load_feature_std_lookup(args.stats, ALTITUDE_FEATURE_NAME)

        args.output_folder.mkdir(parents=True, exist_ok=True)
        errors = 0
        total_attacked_rows = 0
        for input_csv in input_csvs:
            output_csv = args.output_folder / prefixed_output_name(
                input_csv,
                OUTPUT_FILENAME_PREFIX,
            )
            attack_start_bin = (
                args.attack_start_bin
                if args.attack_start_bin is not None
                else random_attack_start_bin(args.bins, attack_rng)
            )
            attack_duration = (
                args.attack_duration
                if args.attack_duration is not None
                else random_attack_duration(attack_rng)
            )
            attack_direction = (
                args.direction
                if args.direction is not None
                else random_attack_direction(attack_rng)
            )
            try:
                attacked_rows, actual_envelope_type = generate_scalar_feature_attack(
                    input_csv=input_csv,
                    output_csv=output_csv,
                    feature_name=ALTITUDE_FEATURE_NAME,
                    attack_type=ATTACK_TYPE,
                    delta_column="attack_delta_altitude",
                    std_column="attack_std_altitude",
                    offset_column="altitude_offset",
                    std_lookup=std_lookup,
                    bins=args.bins,
                    attack_start_bin=attack_start_bin,
                    attack_duration=attack_duration,
                    k=args.k,
                    alpha=args.alpha,
                    direction=attack_direction,
                    label=args.label,
                    duration_amplitude_beta=args.duration_amplitude_beta,
                    envelope_type=args.envelope_type,
                    rng=attack_rng,
                )
                total_attacked_rows += attacked_rows
                print(
                    f"Wrote {output_csv} ({attacked_rows} attacked rows, "
                    f"attack bins {attack_start_bin}-{attack_start_bin + attack_duration}, "
                    f"direction {attack_direction}, envelope {actual_envelope_type})"
                )
            except Exception as exc:
                errors += 1
                print(f"ERROR {input_csv}: {exc}", file=sys.stderr)

        print(f"Input folder: {args.input_folder}")
        print(f"Stats CSV: {args.stats}")
        print(f"Output folder: {args.output_folder}")
        print(f"Selected CSVs: {len(input_csvs)}")
        print(f"Total attacked rows: {total_attacked_rows}")
        print(f"Errors: {errors}")
        return 1 if errors else 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
