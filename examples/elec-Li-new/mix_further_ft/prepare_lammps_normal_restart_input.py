from __future__ import annotations

import argparse
import json
from pathlib import Path


def write_lammps_restart_input(
    *,
    path: Path,
    restart_path: Path,
    model_path: Path,
    pair_coeff_elements: list[str],
    temperature: float,
    timestep_fs: float,
    friction_fs_inv: float,
    steps: int,
    steps_mode: str,
    thermo_interval: int,
    dump_interval: int,
    xtc_dump_interval: int,
    seed: int,
    final_data_path: Path,
    dump_path: Path,
    xtc_path: Path,
    restart_interval: int,
    restart_path_1: Path,
    restart_path_2: Path,
) -> None:
    if timestep_fs <= 0.0:
        raise ValueError("timestep_fs must be positive")

    if friction_fs_inv <= 0.0:
        raise ValueError("friction_fs_inv must be positive")

    if steps < 0:
        raise ValueError("steps must be non-negative")

    if steps_mode not in {"total", "additional"}:
        raise ValueError("steps_mode must be 'total' or 'additional'")

    if thermo_interval <= 0:
        raise ValueError("thermo_interval must be positive")

    if dump_interval <= 0:
        raise ValueError("dump_interval must be positive")

    if xtc_dump_interval <= 0:
        raise ValueError("xtc_dump_interval must be positive")

    if restart_interval <= 0:
        raise ValueError("restart_interval must be positive")

    if not pair_coeff_elements:
        raise ValueError("pair_coeff_elements must not be empty")

    timestep_ps = timestep_fs * 0.001
    damping_ps = 1.0 / friction_fs_inv * 0.001

    restart_name = restart_path.name
    model_name = model_path.name
    final_data_name = final_data_path.name
    dump_name = dump_path.name
    xtc_name = xtc_path.name
    restart_name_1 = restart_path_1.name
    restart_name_2 = restart_path_2.name
    pair_coeff = " ".join(pair_coeff_elements)

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        handle.write("units           metal\n")
        handle.write("atom_style      atomic\n")
        handle.write("boundary        p p p\n\n")

        handle.write(f"variable        RESTART_PATH string {restart_name}\n")
        handle.write(f"variable        MODEL_PATH string {model_name}\n\n")

        handle.write("newton          on\n\n")

        handle.write("read_restart    ${RESTART_PATH}\n\n")

        handle.write("variable        restart_step equal step\n")
        handle.write('print           "Restart timestep = ${restart_step}"\n\n')

        handle.write("pair_style      mliap unified ${MODEL_PATH}\n")
        handle.write(f"pair_coeff      * * {pair_coeff}\n\n")

        handle.write(f"timestep        {timestep_ps:.12g}\n")

        handle.write("fix             int all nve\n")

        handle.write(
            f"fix             therm all temp/csvr "
            f"{temperature:.12g} "
            f"{temperature:.12g} "
            f"{damping_ps:.12g} "
            f"{seed + 7919}\n\n"
        )

        handle.write("neighbor        2.0 bin\n")
        handle.write("neigh_modify    every 1 delay 0 check yes\n\n")

        handle.write(f"thermo          {thermo_interval}\n")
        handle.write(
            "thermo_style    custom "
            "step temp pe ke etotal press\n"
        )
        handle.write("thermo_modify   flush yes\n\n")

        handle.write(
            f"restart         {restart_interval} "
            f"{restart_name_1} "
            f"{restart_name_2}\n\n"
        )

        if steps > 0:
            handle.write(
                f"dump            traj all custom "
                f"{dump_interval} "
                f"{dump_name} "
                "id type x y z\n"
            )

            handle.write("dump_modify     traj sort id\n")

            handle.write(
                f"dump            xtc_traj all xtc "
                f"{xtc_dump_interval} "
                f"{xtc_name}\n"
            )

            handle.write("dump_modify     xtc_traj sort id\n\n")

            if steps_mode == "total":
                handle.write(f"run             {steps} upto\n")
            else:
                handle.write(f"run             {steps}\n")

            handle.write("\n")
            handle.write("undump          traj\n")
            handle.write("undump          xtc_traj\n\n")

        else:
            handle.write("run             0\n\n")

        handle.write(f"write_data      {final_data_name}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a LAMMPS input for continuing normal "
            "MatterTune-MatterSim MLMD from a restart file."
        )
    )

    parser.add_argument(
        "--restart",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--source-metadata",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--model",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=298.15,
    )

    parser.add_argument(
        "--timestep-fs",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--friction-fs-inv",
        type=float,
        default=0.02,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=100000,
    )

    parser.add_argument(
        "--steps-mode",
        choices=("total", "additional"),
        default="total",
    )

    parser.add_argument(
        "--thermo-interval",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--dump-interval",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--xtc-dump-interval",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=7,
    )

    parser.add_argument(
        "--final-data",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--dump",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--xtc-dump",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--restart-interval",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--restart-path-1",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--restart-path-2",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.restart.is_file():
        raise FileNotFoundError(
            f"Restart file not found: {args.restart}"
        )

    if not args.source_metadata.is_file():
        raise FileNotFoundError(
            f"Source metadata not found: {args.source_metadata}"
        )

    source_metadata = json.loads(
        args.source_metadata.read_text(
            encoding="utf-8"
        )
    )

    pair_coeff_elements = source_metadata.get(
        "pair_coeff_elements"
    )

    if not isinstance(
        pair_coeff_elements,
        (list, tuple),
    ):
        raise ValueError(
            f"{args.source_metadata} does not contain "
            "pair_coeff_elements"
        )

    pair_coeff_elements = list(
        pair_coeff_elements
    )

    if not all(
        isinstance(item, str)
        for item in pair_coeff_elements
    ):
        raise ValueError(
            "pair_coeff_elements must contain strings"
        )

    write_lammps_restart_input(
        path=args.input,
        restart_path=args.restart,
        model_path=args.model,
        pair_coeff_elements=pair_coeff_elements,
        temperature=args.temperature,
        timestep_fs=args.timestep_fs,
        friction_fs_inv=args.friction_fs_inv,
        steps=args.steps,
        steps_mode=args.steps_mode,
        thermo_interval=args.thermo_interval,
        dump_interval=args.dump_interval,
        xtc_dump_interval=args.xtc_dump_interval,
        seed=args.seed,
        final_data_path=args.final_data,
        dump_path=args.dump,
        xtc_path=args.xtc_dump,
        restart_interval=args.restart_interval,
        restart_path_1=args.restart_path_1,
        restart_path_2=args.restart_path_2,
    )

    payload = {
        "restart": str(args.restart),
        "source_metadata": str(
            args.source_metadata
        ),
        "input": str(args.input),
        "model": str(args.model),
        "temperature": args.temperature,
        "timestep_fs": args.timestep_fs,
        "friction_fs_inv": (
            args.friction_fs_inv
        ),
        "steps": args.steps,
        "steps_mode": args.steps_mode,
        "thermo_interval": (
            args.thermo_interval
        ),
        "dump_interval": (
            args.dump_interval
        ),
        "xtc_dump_interval": (
            args.xtc_dump_interval
        ),
        "seed": args.seed,
        "final_data": str(
            args.final_data
        ),
        "dump": str(args.dump),
        "xtc_dump": str(args.xtc_dump),
        "restart_interval": (
            args.restart_interval
        ),
        "restart_path_1": str(
            args.restart_path_1
        ),
        "restart_path_2": str(
            args.restart_path_2
        ),
        "pair_coeff_elements": (
            pair_coeff_elements
        ),
    }

    args.metadata.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.metadata.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"restart={args.restart}")
    print(
        f"source_metadata="
        f"{args.source_metadata}"
    )
    print(f"steps={args.steps}")
    print(
        f"steps_mode="
        f"{args.steps_mode}"
    )
    print(
        f"restart_interval="
        f"{args.restart_interval}"
    )
    print(
        f"restart_path_1="
        f"{args.restart_path_1}"
    )
    print(
        f"restart_path_2="
        f"{args.restart_path_2}"
    )
    print(f"wrote {args.input}")
    print(f"wrote {args.metadata}")
    print(
        "pair_coeff * * "
        + " ".join(
            pair_coeff_elements
        )
    )


if __name__ == "__main__":
    main()
