from __future__ import annotations

import argparse
import csv
import fcntl
import math
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


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


def format_float(
    value: str,
    *,
    is_time: bool = False,
) -> str:
    if value == "":
        return ""

    number = float(value)

    if math.isnan(number):
        return ""

    if is_time:
        return f"{number:.12g}"

    return f"{number:.16g}"


@contextmanager
def exclusive_lock(path: Path) -> Iterator[None]:
    """
    Serialize periodic/final/recovery merges for one lambda directory.
    """
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "a+",
        encoding="utf-8",
    ) as handle:
        fcntl.flock(
            handle.fileno(),
            fcntl.LOCK_EX,
        )

        try:
            yield
        finally:
            fcntl.flock(
                handle.fileno(),
                fcntl.LOCK_UN,
            )


def read_energy_rows(
    path: Path,
    *,
    tolerate_incomplete_tail: bool = False,
) -> list[dict[str, str]]:
    """
    Read an energy CSV.

    During MD the final CSV line may be in the middle of being written while
    a periodic refresh reads the file.  In periodic mode that incomplete
    trailing row is ignored and will be picked up on the next refresh.
    """
    if not path.is_file():
        return []

    rows: list[dict[str, str]] = []

    with path.open(
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            return []

        if tuple(reader.fieldnames) != ENERGY_LOG_FIELDS:
            raise ValueError(
                f"Unexpected columns in {path}: "
                f"{reader.fieldnames}"
            )

        for row in reader:
            incomplete = any(
                row.get(field) is None
                for field in ENERGY_LOG_FIELDS
            )

            if incomplete:
                if tolerate_incomplete_tail:
                    continue

                raise ValueError(
                    f"Incomplete energy row in {path}: {row}"
                )

            step_text = row.get("step", "")

            if not step_text.strip():
                if tolerate_incomplete_tail:
                    continue

                raise ValueError(
                    f"Missing step in energy row from {path}"
                )

            try:
                int(float(step_text))
            except ValueError:
                if tolerate_incomplete_tail:
                    continue
                raise

            rows.append(
                {
                    field: row.get(field, "")
                    for field in ENERGY_LOG_FIELDS
                }
            )

    return rows


def write_energy_rows_atomic(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp_path = path.with_name(
        path.name + ".tmp"
    )

    with tmp_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=ENERGY_LOG_FIELDS,
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())

    os.replace(
        tmp_path,
        path,
    )


def read_restart_step(
    path: Path,
) -> int:
    if not path.is_file():
        raise FileNotFoundError(
            f"restart step file does not exist: {path}"
        )

    text = path.read_text(
        encoding="utf-8"
    ).strip()

    if not text:
        raise ValueError(
            f"restart step file is empty: {path}"
        )

    return int(
        float(
            text.splitlines()[-1].strip()
        )
    )


def append_restart_segment(
    *,
    destination: Path,
    segment: Path,
    restart_step: int,
    timestep_fs: float,
    keep_segment: bool,
    quiet: bool,
) -> None:
    """
    Merge a restart continuation into the canonical raw trajectory.

    Rows already present at or after restart_step belong to an older
    continuation branch and are replaced by the current restart segment.

    When keep_segment=True the currently active segment is left in place
    because LAMMPS/ML-IAP is still writing to it.  This is used by the
    2-minute periodic refresh.

    When keep_segment=False the segment is finalized and removed after a
    successful merge.
    """
    if not segment.is_file():
        raise FileNotFoundError(segment)

    existing_rows = read_energy_rows(
        destination,
        tolerate_incomplete_tail=True,
    )

    segment_rows = read_energy_rows(
        segment,
        tolerate_incomplete_tail=True,
    )

    if not segment_rows:
        if not keep_segment:
            segment.unlink(
                missing_ok=True
            )

        if not quiet:
            if keep_segment:
                print(
                    f"restart segment currently empty: "
                    f"{segment}"
                )
            else:
                print(
                    f"empty segment removed: {segment}"
                )

        return

    preserved_rows: list[
        dict[str, str]
    ] = []

    for row in existing_rows:
        step = int(
            float(row["step"])
        )

        if step < restart_step:
            preserved_rows.append(row)

    converted_rows: list[
        dict[str, str]
    ] = []

    for source_row in segment_rows:
        row = dict(source_row)

        local_step = int(
            float(row["step"])
        )

        absolute_step = (
            restart_step + local_step
        )

        absolute_time_fs = (
            absolute_step
            * timestep_fs
        )

        absolute_time_ps = (
            absolute_time_fs
            * 0.001
        )

        row["step"] = str(
            absolute_step
        )

        row["time_fs"] = (
            f"{absolute_time_fs:.12g}"
        )

        row["time_ps"] = (
            f"{absolute_time_ps:.12g}"
        )

        converted_rows.append(row)

    by_step: dict[
        int,
        dict[str, str],
    ] = {}

    for row in preserved_rows:
        step = int(
            float(row["step"])
        )

        by_step[step] = row

    for row in converted_rows:
        step = int(
            float(row["step"])
        )

        by_step[step] = row

    final_rows = [
        by_step[step]
        for step in sorted(by_step)
    ]

    write_energy_rows_atomic(
        destination,
        final_rows,
    )

    if not keep_segment:
        segment.unlink(
            missing_ok=True
        )

    if not quiet:
        mode = (
            "periodic snapshot"
            if keep_segment
            else "final"
        )

        print(
            f"merged restart segment "
            f"({mode}): "
            f"restart_step={restart_step} "
            f"preserved={len(preserved_rows)} "
            f"segment={len(converted_rows)} "
            f"total={len(final_rows)}"
        )

        print(
            f"updated {destination}"
        )


def read_temperature_by_step(
    path: Path,
) -> dict[int, str]:
    values: dict[int, str] = {}

    if not path.is_file():
        return values

    with path.open(
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.reader(handle)

        for row in reader:
            if len(row) < 2:
                continue

            first = row[0].strip()
            second = row[1].strip()

            try:
                step = int(
                    float(first)
                )

                temperature = (
                    format_float(second)
                )

            except ValueError:
                continue

            # Restart temperature files may contain a step more than once.
            # The newest occurrence is the canonical continuation.
            values[step] = temperature

    return values


def merge_logs(
    *,
    energy_log_raw: Path,
    temperature_log: Path,
    output: Path,
    quiet: bool,
) -> None:
    rows = read_energy_rows(
        energy_log_raw,
        tolerate_incomplete_tail=True,
    )

    temperature_by_step = (
        read_temperature_by_step(
            temperature_log
        )
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp_path = output.with_name(
        output.name + ".tmp"
    )

    matched_temperatures = 0

    with tmp_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=ENERGY_LOG_FIELDS,
            lineterminator="\n",
        )

        writer.writeheader()

        for row in rows:
            merged: dict[str, str] = {}

            step = int(
                float(row["step"])
            )

            for field in ENERGY_LOG_FIELDS:
                value = row.get(
                    field,
                    "",
                )

                if field == "temperature_K":
                    if step in temperature_by_step:
                        value = (
                            temperature_by_step[
                                step
                            ]
                        )

                        matched_temperatures += 1

                    else:
                        value = ""

                elif field in FLOAT_FIELDS:
                    value = format_float(
                        value,
                        is_time=(
                            field
                            in TIME_FIELDS
                        ),
                    )

                merged[field] = value

            writer.writerow(merged)

        handle.flush()
        os.fsync(handle.fileno())

    os.replace(
        tmp_path,
        output,
    )

    if not quiet:
        print(
            f"wrote {output} "
            f"rows={len(rows)} "
            f"temperature_matches="
            f"{matched_temperatures}"
        )

        if (
            matched_temperatures
            < len(rows)
        ):
            print(
                "warning: some energy steps "
                "have no matching temperature "
                "row; temperature_K was left "
                "empty for those steps."
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge ML-IAP FEP-TI energy data "
            "with LAMMPS temperature data."
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

    parser.add_argument(
        "--keep-segment",
        action="store_true",
        help=(
            "Merge a restart segment snapshot "
            "without deleting the active segment."
        ),
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    lock_path = (
        args.energy_log_raw.parent
        / ".fep-ti-energy.merge.lock"
    )

    with exclusive_lock(lock_path):
        if (
            args.append_raw_segment
            is not None
        ):
            restart_step = (
                args.restart_step
            )

            if (
                restart_step is None
                and args.restart_step_file
                is not None
            ):
                restart_step = (
                    read_restart_step(
                        args.restart_step_file
                    )
                )

            if restart_step is None:
                raise ValueError(
                    "--restart-step or "
                    "--restart-step-file is "
                    "required when "
                    "--append-raw-segment "
                    "is used"
                )

            if args.timestep_fs is None:
                raise ValueError(
                    "--timestep-fs is required "
                    "when --append-raw-segment "
                    "is used"
                )

            append_restart_segment(
                destination=(
                    args.energy_log_raw
                ),
                segment=(
                    args.append_raw_segment
                ),
                restart_step=restart_step,
                timestep_fs=(
                    args.timestep_fs
                ),
                keep_segment=(
                    args.keep_segment
                ),
                quiet=args.quiet,
            )

        merge_logs(
            energy_log_raw=(
                args.energy_log_raw
            ),
            temperature_log=(
                args.temperature_log
            ),
            output=args.output,
            quiet=args.quiet,
        )


if __name__ == "__main__":
    main()
