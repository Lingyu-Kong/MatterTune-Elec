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
    temperature_log_path: Path | None,
    temperature_log_interval: int,
    restart_interval: int,
    restart_path_1: Path,
    restart_path_2: Path,
    restart_step_path: Path,
) -> None:
    if timestep_fs <= 0.0:
        raise ValueError("timestep_fs must be positive")

    if friction_fs_inv <= 0.0:
        raise ValueError("friction_fs_inv must be positive")

    if steps < 0:
        raise ValueError("steps must be non-negative")

    if steps_mode not in {"additional", "total"}:
        raise ValueError(
            "steps_mode must be 'additional' or 'total'"
        )

    if min(
        thermo_interval,
        dump_interval,
        xtc_dump_interval,
        restart_interval,
    ) <= 0:
        raise ValueError(
            "output and restart intervals must be positive"
        )

    if (
        temperature_log_path is not None
        and temperature_log_interval <= 0
    ):
        raise ValueError(
            "temperature_log_interval must be positive"
        )

    if not pair_coeff_elements:
        raise ValueError(
            "pair_coeff_elements must not be empty"
        )

    timestep_ps = timestep_fs * 0.001
    damping_ps = 1.0 / friction_fs_inv * 0.001
    pair_coeff = " ".join(pair_coeff_elements)

    #
    # IMPORTANT:
    # Every file below belongs to the current lambda run directory.
    #
    # LAMMPS will be launched from that directory, so the generated
    # input must contain basenames only. This makes the input portable
    # between /projects, /oscar, or any other cluster filesystem.
    #
    restart_name = restart_path.name
    model_name = model_path.name
    final_data_name = final_data_path.name
    dump_name = dump_path.name
    xtc_name = xtc_path.name
    restart_name_1 = restart_path_1.name
    restart_name_2 = restart_path_2.name
    restart_step_name = restart_step_path.name

    temperature_log_name = (
        None
        if temperature_log_path is None
        else temperature_log_path.name
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(
            "units           metal\n"
        )
        handle.write(
            "atom_style      atomic\n"
        )
        handle.write(
            "boundary        p p p\n\n"
        )

        handle.write(
            f"variable        RESTART_PATH string "
            f"{restart_name}\n"
        )
        handle.write(
            f"variable        MODEL_PATH string "
            f"{model_name}\n\n"
        )

        handle.write(
            "newton          on\n\n"
        )

        handle.write(
            "read_restart    ${RESTART_PATH}\n\n"
        )

        #
        # read_restart restores the absolute LAMMPS timestep.
        #
        # Record that timestep before continuation so that the
        # FEP-TI energy segment can later be mapped back to the
        # absolute trajectory step.
        #
        handle.write(
            "variable        fep_ti_restart_step equal step\n"
        )

        handle.write(
            f'print           "${{fep_ti_restart_step}}" '
            f"file {restart_step_name} screen no\n\n"
        )

        handle.write(
            "pair_style      mliap unified ${MODEL_PATH}\n"
        )

        handle.write(
            f"pair_coeff      * * {pair_coeff}\n\n"
        )

        handle.write(
            f"timestep        {timestep_ps:.12g}\n"
        )

        handle.write(
            "fix             int all nve\n"
        )

        handle.write(
            f"fix             therm all temp/csvr "
            f"{temperature:.12g} "
            f"{temperature:.12g} "
            f"{damping_ps:.12g} "
            f"{seed + 7919}\n\n"
        )

        handle.write(
            "neighbor        2.0 bin\n"
        )

        handle.write(
            "neigh_modify    every 1 delay 0 check yes\n\n"
        )

        if temperature_log_name is not None:
            handle.write(
                "variable        fep_ti_log_step equal step\n"
            )

            handle.write(
                "variable        fep_ti_log_temp equal temp\n"
            )

            handle.write(
                f"fix             fep_ti_temp_log all print "
                f"{temperature_log_interval} "
                f'"${{fep_ti_log_step}},'
                f'${{fep_ti_log_temp}}" '
                f"append {temperature_log_name} "
                'screen no title ""\n\n'
            )

        handle.write(
            f"thermo          {thermo_interval}\n"
        )

        handle.write(
            "thermo_style    custom "
            "step temp pe ke etotal press\n"
        )

        handle.write(
            "thermo_modify   flush yes\n\n"
        )

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

            handle.write(
                "dump_modify     traj sort id\n"
            )

            handle.write(
                f"dump            xtc_traj all xtc "
                f"{xtc_dump_interval} "
                f"{xtc_name}\n"
            )

            handle.write(
                "dump_modify     xtc_traj sort id\n"
            )

            if steps_mode == "total":
                handle.write(
                    f"run             {steps} upto\n"
                )
            else:
                handle.write(
                    f"run             {steps}\n"
                )

            handle.write(
                "undump          traj\n"
            )

            handle.write(
                "undump          xtc_traj\n"
            )

        else:
            handle.write(
                "run             0\n"
            )

        handle.write(
            f"write_data      {final_data_name}\n"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a portable LAMMPS input that continues "
            "ghost-target FEP-TI from a restart file."
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
        "--restart-step-file",
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
        choices=(
            "additional",
            "total",
        ),
        default="total",
        help=(
            "total: run until the absolute production timestep "
            "reaches --steps; "
            "additional: run --steps more steps from the restart."
        ),
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
        "--temperature-log",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--temperature-log-interval",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--restart-interval",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--restart-path-1",
        type=Path,
        default=Path("lmp.restart.1"),
    )

    parser.add_argument(
        "--restart-path-2",
        type=Path,
        default=Path("lmp.restart.2"),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source_metadata = json.loads(
        args.source_metadata.read_text(
            encoding="utf-8"
        )
    )

    pair_coeff_elements = source_metadata.get(
        "pair_coeff_elements"
    )

    if (
        not isinstance(
            pair_coeff_elements,
            list,
        )
        or not all(
            isinstance(item, str)
            for item in pair_coeff_elements
        )
    ):
        raise ValueError(
            f"{args.source_metadata} does not contain "
            "a valid pair_coeff_elements list"
        )

    #
    # IMPORTANT:
    #
    # Do not call Path.resolve() here.
    #
    # The shell orchestration is allowed to pass absolute paths,
    # but write_lammps_restart_input() converts every run-local
    # reference to Path.name before writing the LAMMPS input.
    #
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
        temperature_log_path=args.temperature_log,
        temperature_log_interval=(
            args.temperature_log_interval
        ),
        restart_interval=args.restart_interval,
        restart_path_1=args.restart_path_1,
        restart_path_2=args.restart_path_2,
        restart_step_path=args.restart_step_file,
    )

    #
    # Metadata can retain the paths supplied to Python.
    # These values are orchestration/debugging information and are
    # not embedded into the LAMMPS input itself.
    #
    payload = {
        "restart": str(args.restart),
        "source_metadata": str(
            args.source_metadata
        ),
        "input": str(args.input),
        "model": str(args.model),
        "restart_step_file": str(
            args.restart_step_file
        ),
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
        "dump_interval": args.dump_interval,
        "xtc_dump_interval": (
            args.xtc_dump_interval
        ),
        "seed": args.seed,
        "final_data": str(args.final_data),
        "dump": str(args.dump),
        "xtc_dump": str(args.xtc_dump),
        "temperature_log": (
            None
            if args.temperature_log is None
            else str(args.temperature_log)
        ),
        "temperature_log_interval": (
            args.temperature_log_interval
        ),
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

    print(
        f"wrote {args.input}"
    )

    print(
        f"wrote {args.metadata}"
    )

    print(
        f"restart={args.restart}"
    )

    print(
        f"restart_step_file="
        f"{args.restart_step_file}"
    )

    print(
        f"xtc_dump={args.xtc_dump}"
    )


if __name__ == "__main__":
    main()
