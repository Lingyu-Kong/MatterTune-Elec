from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path


ENERGY_LOG_FIELDS = (
    "step",
    "time_fs",
    "time_ps",
    "temperature_K",
    "mixed_energy_eV",
    "E_I_eV",
    "E_F_with_LJ_eV",
    "E_F_without_LJ_eV",
    "E_LJ_eV",
    "deltaE_with_LJ_eV",
    "deltaE_without_LJ_eV",
)

TIME_FIELDS = {"time_fs", "time_ps"}
FLOAT_FIELDS = set(ENERGY_LOG_FIELDS) - {"step"}


def format_float(value: str, *, is_time: bool = False) -> str:
    if value == "":
        return ""
    number = float(value)
    if math.isnan(number):
        return ""
    return f"{number:.12g}" if is_time else f"{number:.16g}"


def read_energy_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            return []

        if tuple(reader.fieldnames) != ENERGY_LOG_FIELDS:
            raise ValueError(
                f"Unexpected columns in {path}: {reader.fieldnames}"
            )

        return list(reader)


def write_energy_rows_atomic(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")

    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=ENERGY_LOG_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    os.replace(tmp_path, path)


def append_restart_segment(
    *,
    destination: Path,
    segment: Path,
    restart_step: int,
    timestep_fs: float,
) -> None:
    if not segment.is_file():
        raise FileNotFoundError(segment)

    existing_rows = read_energy_rows(destination)
    segment_rows = read_energy_rows(segment)

    if not segment_rows:
        segment.unlink()
        print(f"empty segment removed: {segment}")
        return

    # A restart defines a new canonical trajectory beginning at
    # restart_step. Anything already present at or after this step belongs
    # to an older continuation branch and must not be retained.
    preserved_rows: list[dict[str, str]] = []

    for row in existing_rows:
        step = int(float(row["step"]))
        if step < restart_step:
            preserved_rows.append(row)

    converted_rows: list[dict[str, str]] = []

    for source_row in segment_rows:
        row = dict(source_row)

        local_step = int(float(row["step"]))
        absolute_step = restart_step + local_step

        absolute_time_fs = absolute_step * timestep_fs
        absolute_time_ps = absolute_time_fs * 0.001

        row["step"] = str(absolute_step)
        row["time_fs"] = f"{absolute_time_fs:.12g}"
        row["time_ps"] = f"{absolute_time_ps:.12g}"

        converted_rows.append(row)

    # Protect against an unexpected duplicate inside the segment itself.
    by_step: dict[int, dict[str, str]] = {}

    for row in preserved_rows:
        by_step[int(float(row["step"]))] = row

    for row in converted_rows:
        by_step[int(float(row["step"]))] = row

    final_rows = [by_step[step] for step in sorted(by_step)]

    write_energy_rows_atomic(destination, final_rows)

    segment.unlink()

    print(
        f"merged restart segment: restart_step={restart_step} "
        f"preserved={len(preserved_rows)} "
        f"segment={len(converted_rows)} "
        f"total={len(final_rows)}"
    )
    print(f"updated {destination}")


def read_temperature_by_step(path: Path) -> dict[int, str]:
    values: dict[int, str] = {}

    if not path.is_file():
        return values

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)

        for row in reader:
            if len(row) < 2:
                continue

            first = row[0].strip()
            second = row[1].strip()

            # Support either:
            # step,temperature_K
            # or headerless files written by fix print.
            try:
                step = int(float(first))
                temperature = format_float(second)
            except ValueError:
                continue

            # Restarted runs may append the same absolute step again.
            # The newest occurrence is the canonical continuation.
            values[step] = temperature

    return values


def merge_logs(
    *,
    energy_log_raw: Path,
    temperature_log: Path,
    output: Path,
) -> None:
    rows = read_energy_rows(energy_log_raw)
    temperature_by_step = read_temperature_by_step(temperature_log)

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output.with_name(output.name + ".tmp")

    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=ENERGY_LOG_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()

        matched_temperatures = 0

        for row in rows:
            merged: dict[str, str] = {}
            step = int(float(row["step"]))

            for field in ENERGY_LOG_FIELDS:
                value = row.get(field, "")

                if field == "temperature_K":
                    if step in temperature_by_step:
                        value = temperature_by_step[step]
                        matched_temperatures += 1
                    else:
                        value = ""

                elif field in FLOAT_FIELDS:
                    value = format_float(
                        value,
                        is_time=field in TIME_FIELDS,
                    )

                merged[field] = value

            writer.writerow(merged)

    os.replace(tmp_path, output)

    print(
        f"wrote {output} rows={len(rows)} "
        f"temperature_matches={matched_temperatures}"
    )

    if matched_temperatures < len(rows):
        print(
            "warning: some energy steps have no matching temperature row; "
            "temperature_K was left empty for those steps."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge ML-IAP FEP-TI energy data with LAMMPS temperature data."
        )
    )

    parser.add_argument(
        "--energy-log-raw",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--temperature-log",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--append-raw-segment",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--restart-step",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--restart-step-file",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--timestep-fs",
        type=float,
        default=None,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.append_raw_segment is not None:
        restart_step = args.restart_step

        if restart_step is None and args.restart_step_file is not None:
            if not args.restart_step_file.is_file():
                raise FileNotFoundError(
                    f"restart step file does not exist: "
                    f"{args.restart_step_file}"
                )

            text = args.restart_step_file.read_text(
                encoding="utf-8"
            ).strip()

            if not text:
                raise ValueError(
                    f"restart step file is empty: "
                    f"{args.restart_step_file}"
                )

            restart_step = int(float(text.splitlines()[-1].strip()))

        if restart_step is None:
            raise ValueError(
                "--restart-step or --restart-step-file is required "
                "when --append-raw-segment is used"
            )

        if args.timestep_fs is None:
            raise ValueError(
                "--timestep-fs is required when "
                "--append-raw-segment is used"
            )

        append_restart_segment(
            destination=args.energy_log_raw,
            segment=args.append_raw_segment,
            restart_step=restart_step,
            timestep_fs=args.timestep_fs,
        )

    merge_logs(
        energy_log_raw=args.energy_log_raw,
        temperature_log=args.temperature_log,
        output=args.output,
    )


if __name__ == "__main__":
    main()
