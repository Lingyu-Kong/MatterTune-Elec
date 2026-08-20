from __future__ import annotations

import argparse
import csv
import math
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


def read_temperature_values(path: Path) -> list[str]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return []
        if "temperature_K" not in reader.fieldnames:
            raise ValueError(f"{path} does not contain a temperature_K column.")
        return [format_float(row.get("temperature_K", "")) for row in reader]


def append_energy_segment(*, destination: Path, segment: Path) -> None:
    """Append one restart energy segment while keeping step/time columns continuous."""
    if not segment.is_file():
        raise FileNotFoundError(segment)

    existing_rows: list[dict[str, str]] = []
    if destination.is_file():
        with destination.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"{destination} is empty.")
            if tuple(reader.fieldnames) != ENERGY_LOG_FIELDS:
                raise ValueError(f"Unexpected columns in {destination}: {reader.fieldnames}")
            existing_rows = list(reader)

    with segment.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{segment} is empty.")
        if tuple(reader.fieldnames) != ENERGY_LOG_FIELDS:
            raise ValueError(f"Unexpected columns in {segment}: {reader.fieldnames}")
        segment_rows = list(reader)

    if not segment_rows:
        segment.unlink()
        return

    step_offset = 0
    time_fs_offset = 0.0
    time_ps_offset = 0.0
    if existing_rows:
        previous = existing_rows[-1]
        first = segment_rows[0]
        if len(segment_rows) > 1:
            step_stride = int(float(segment_rows[1]["step"])) - int(float(first["step"]))
            time_fs_stride = float(segment_rows[1]["time_fs"]) - float(first["time_fs"])
            time_ps_stride = float(segment_rows[1]["time_ps"]) - float(first["time_ps"])
        elif len(existing_rows) > 1:
            step_stride = int(float(previous["step"])) - int(float(existing_rows[-2]["step"]))
            time_fs_stride = float(previous["time_fs"]) - float(existing_rows[-2]["time_fs"])
            time_ps_stride = float(previous["time_ps"]) - float(existing_rows[-2]["time_ps"])
        else:
            step_stride = 1
            time_fs_stride = 0.0
            time_ps_stride = 0.0
        step_offset = int(float(previous["step"])) + step_stride - int(float(first["step"]))
        time_fs_offset = float(previous["time_fs"]) + time_fs_stride - float(first["time_fs"])
        time_ps_offset = float(previous["time_ps"]) + time_ps_stride - float(first["time_ps"])

    destination.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if existing_rows else "w"
    with destination.open(mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ENERGY_LOG_FIELDS, lineterminator="\n")
        if not existing_rows:
            writer.writeheader()
        for row in segment_rows:
            row["step"] = str(int(float(row["step"])) + step_offset)
            row["time_fs"] = f"{float(row['time_fs']) + time_fs_offset:.12g}"
            row["time_ps"] = f"{float(row['time_ps']) + time_ps_offset:.12g}"
            writer.writerow(row)

    segment.unlink()
    print(f"appended {len(segment_rows)} rows from {segment} to {destination}")


def merge_logs(*, energy_log_raw: Path, temperature_log: Path, output: Path) -> None:
    with energy_log_raw.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{energy_log_raw} is empty.")
        missing = set(ENERGY_LOG_FIELDS) - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"{energy_log_raw} is missing columns: {', '.join(sorted(missing))}"
            )
        rows = list(reader)

    temperature_values = read_temperature_values(temperature_log)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=ENERGY_LOG_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        for index, row in enumerate(rows):
            merged: dict[str, str] = {}
            for field in ENERGY_LOG_FIELDS:
                value = row.get(field, "")
                if field == "temperature_K":
                    value = (
                        temperature_values[index]
                        if index < len(temperature_values)
                        else ""
                    )
                elif field in FLOAT_FIELDS:
                    value = format_float(value, is_time=field in TIME_FIELDS)
                merged[field] = value
            writer.writerow(merged)

    print(
        f"wrote {output} rows={len(rows)} "
        f"temperature_rows={len(temperature_values)}"
    )
    if len(temperature_values) < len(rows):
        print(
            "warning: fewer temperature rows than energy rows; missing temperatures "
            "were written as empty fields."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge the ML-IAP FEP-TI energy CSV with the per-step LAMMPS "
            "temperature sidecar."
        )
    )
    parser.add_argument("--energy-log-raw", type=Path, required=True)
    parser.add_argument("--temperature-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--append-raw-segment",
        type=Path,
        default=None,
        help="Append this restart raw-energy segment before rebuilding the merged log.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.append_raw_segment is not None:
        append_energy_segment(
            destination=args.energy_log_raw,
            segment=args.append_raw_segment,
        )
    merge_logs(
        energy_log_raw=args.energy_log_raw,
        temperature_log=args.temperature_log,
        output=args.output,
    )


if __name__ == "__main__":
    main()
