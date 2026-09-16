from __future__ import annotations
import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ELEMENT_ORDER = ("Li", "F", "S", "N", "O", "C", "H")

DEFAULT_MASSES = {
    "Li": 6.94,
    "F": 18.998403163,
    "S": 32.06,
    "N": 14.007,
    "O": 15.999,
    "C": 12.011,
    "H": 1.008,
}

@dataclass(frozen=True)
class PDBAtom:
    index: int
    serial: int
    name: str
    residue: str
    residue_id: int
    element: str
    x: float
    y: float
    z: float

def parse_element_order(value: str) -> tuple[str, ...]:
    order = tuple(piece.strip() for piece in value.split(",") if piece.strip())
    if not order:
        raise argparse.ArgumentTypeError("--element-order must not be empty")
    if len(set(order)) != len(order):
        raise argparse.ArgumentTypeError("--element-order must not contain duplicates")
    for element in order:
        if element not in DEFAULT_MASSES:
            raise argparse.ArgumentTypeError(f"Unsupported element: {element}")
    return order

def element_from_pdb_atom_name(name: str) -> str:
    letters = "".join(ch for ch in name.strip() if ch.isalpha())
    if not letters:
        raise ValueError(f"Cannot infer element from atom name {name!r}")
    upper = letters.upper()
    if upper.startswith("LI"):
        return "Li"
    element = upper[0]
    if element not in DEFAULT_MASSES:
        raise ValueError(f"Unsupported element inferred from {name!r}: {element}")
    return element

def parse_cryst1(line: str) -> tuple[float, float, float, float, float, float]:
    fields = line.split()
    if len(fields) < 7:
        raise ValueError(f"Malformed CRYST1 record: {line.rstrip()}")
    return tuple(float(value) for value in fields[1:7])

def parse_pdb(path: Path) -> tuple[list[PDBAtom], tuple[float, float, float]]:
    atoms = []
    cell = None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = line[:6].strip()
            if record == "CRYST1":
                a, b, c, alpha, beta, gamma = parse_cryst1(line)
                if any(abs(angle - 90.0) > 1e-6 for angle in (alpha, beta, gamma)):
                    raise ValueError(f"Only orthorhombic cells are supported: {(alpha, beta, gamma)}")
                cell = (a, b, c)
            elif record in {"ATOM", "HETATM"}:
                name = line[12:16].strip()
                residue = line[17:20].strip()
                residue_id_text = line[22:26].strip()
                atoms.append(
                    PDBAtom(
                        index=len(atoms),
                        serial=int(line[6:11]),
                        name=name,
                        residue=residue,
                        residue_id=int(residue_id_text) if residue_id_text else 0,
                        element=element_from_pdb_atom_name(name),
                        x=float(line[30:38]),
                        y=float(line[38:46]),
                        z=float(line[46:54]),
                    )
                )
    if cell is None:
        raise ValueError(f"No CRYST1 record found in {path}")
    if not atoms:
        raise ValueError(f"No atoms found in {path}")
    return atoms, cell

def wrap_position(value: float, length: float) -> float:
    wrapped = math.fmod(value, length)
    if wrapped < 0.0:
        wrapped += length
    return wrapped

def write_lammps_data(
    *,
    path: Path,
    atoms: list[PDBAtom],
    cell: tuple[float, float, float],
    element_order: tuple[str, ...],
) -> dict[str, object]:
    base_types = {element: i + 1 for i, element in enumerate(element_order)}
    for atom in atoms:
        if atom.element not in base_types:
            raise ValueError(f"Atom index {atom.index} has element {atom.element}, not in {element_order}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("LAMMPS data generated from PDB for normal MatterTune-MatterSim MD\n\n")
        handle.write(f"{len(atoms)} atoms\n")
        handle.write(f"{len(element_order)} atom types\n\n")
        handle.write(f"0.0 {cell[0]:.10f} xlo xhi\n")
        handle.write(f"0.0 {cell[1]:.10f} ylo yhi\n")
        handle.write(f"0.0 {cell[2]:.10f} zlo zhi\n\n")
        handle.write("Masses\n\n")
        for atom_type, element in enumerate(element_order, start=1):
            handle.write(f"{atom_type} {DEFAULT_MASSES[element]:.12g} # {element}\n")
        handle.write("\nAtoms # atomic\n\n")
        for atom in atoms:
            x = wrap_position(atom.x, cell[0])
            y = wrap_position(atom.y, cell[1])
            z = wrap_position(atom.z, cell[2])
            handle.write(
                f"{atom.index + 1} {base_types[atom.element]} "
                f"{x:.10f} {y:.10f} {z:.10f} "
                f"# pdb_index={atom.index} pdb_serial={atom.serial} "
                f"name={atom.name} residue={atom.residue}{atom.residue_id}\n"
            )
    return {
        "natoms": len(atoms),
        "cell": cell,
        "element_order": element_order,
        "base_types": base_types,
        "pair_coeff_elements": element_order,
    }

def write_lammps_input(
    *,
    path: Path,
    data_path: Path,
    model_path: Path,
    pair_coeff_elements: list[str],
    temperature: float,
    timestep_fs: float,
    friction_fs_inv: float,
    warmup_steps: int,
    steps: int,
    thermo_interval: int,
    dump_interval: int,
    xtc_dump_interval: int,
    xtc_path: Path,
    seed: int,
    init_velocities: bool,
    final_data_path: Path,
    dump_path: Path,
    restart_interval: int,
    restart_path_1: Path,
    restart_path_2: Path,
    final_restart_path: Path,
    restart_from: Path | None = None,
) -> None:
    if timestep_fs <= 0:
        raise ValueError("timestep_fs must be positive")
    if friction_fs_inv <= 0:
        raise ValueError("friction_fs_inv must be positive")
    if warmup_steps < 0 or steps < 0:
        raise ValueError("warmup_steps and steps must be non-negative")
    if thermo_interval <= 0 or dump_interval <= 0 or xtc_dump_interval <= 0:
        raise ValueError("output intervals must be positive")
    if restart_interval <= 0:
        raise ValueError("restart_interval must be positive")

    timestep_ps = timestep_fs * 0.001
    damping_ps = 1.0 / friction_fs_inv * 0.001

    data_name = data_path.name
    model_name = model_path.name
    final_data_name = final_data_path.name
    dump_name = dump_path.name
    xtc_name = xtc_path.name
    restart_name_1 = restart_path_1.name
    restart_name_2 = restart_path_2.name
    final_restart_name = final_restart_path.name

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        handle.write("units           metal\n")
        handle.write("atom_style      atomic\n")
        handle.write("boundary        p p p\n\n")
        handle.write(f"variable        MODEL_PATH string {model_name}\n")

        if restart_from is None:
            handle.write(f"variable        DATA_PATH string {data_name}\n\n")
        else:
            handle.write("\n")

        handle.write("newton          on\n\n")

        if restart_from is None:
            handle.write("read_data       ${DATA_PATH}\n\n")
        else:
            handle.write(f"read_restart    {restart_from.resolve()}\n\n")

        handle.write("pair_style      mliap unified ${MODEL_PATH}\n")
        handle.write(f"pair_coeff      * * {' '.join(pair_coeff_elements)}\n\n")

        if restart_from is None and init_velocities:
            handle.write(
                f"velocity        all create {temperature:.12g} {seed} "
                "mom yes rot yes dist gaussian\n\n"
            )

        handle.write(f"timestep        {timestep_ps:.12g}\n")
        handle.write("fix             int all nve\n")
        handle.write(
            f"fix             therm all temp/csvr "
            f"{temperature:.12g} {temperature:.12g} "
            f"{damping_ps:.12g} {seed + 7919}\n\n"
        )

        handle.write("neighbor        2.0 bin\n")
        handle.write("neigh_modify    every 1 delay 0 check yes\n\n")

        handle.write(f"thermo          {thermo_interval}\n")
        handle.write("thermo_style    custom step temp pe ke etotal press\n")
        handle.write("thermo_modify   flush yes\n\n")

        if restart_from is None and warmup_steps > 0:
            handle.write(f"run             {warmup_steps}\n")
            handle.write("reset_timestep  0\n\n")

        handle.write(
            f"restart         {restart_interval} "
            f"{restart_name_1} {restart_name_2}\n\n"
        )

        if steps > 0:
            handle.write(
                f"dump            traj all custom {dump_interval} "
                f"{dump_name} id type x y z\n"
            )
            handle.write("dump_modify     traj sort id\n")
            handle.write(
                f"dump            xtc_traj all xtc "
                f"{xtc_dump_interval} {xtc_name}\n"
            )
            handle.write("dump_modify     xtc_traj sort id\n\n")
            handle.write(f"run             {steps}\n\n")
            handle.write("undump          traj\n")
            handle.write("undump          xtc_traj\n\n")
        else:
            handle.write("run             0\n\n")

        handle.write(f"write_data      {final_data_name}\n")
        handle.write(f"write_restart   {final_restart_name}\n")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--element-order", type=parse_element_order, default=DEFAULT_ELEMENT_ORDER)
    parser.add_argument("--temperature", type=float, default=298.15)
    parser.add_argument("--timestep-fs", type=float, default=1.0)
    parser.add_argument("--friction-fs-inv", type=float, default=0.02)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--thermo-interval", type=int, default=100)
    parser.add_argument("--dump-interval", type=int, default=100)
    parser.add_argument("--xtc-dump-interval", type=int, default=500)
    parser.add_argument("--xtc-path", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--init-velocities", dest="init_velocities", action="store_true", default=True)
    parser.add_argument("--no-init-velocities", dest="init_velocities", action="store_false")
    parser.add_argument("--final-data", type=Path, required=True)
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--restart-interval", type=int, default=10000)
    parser.add_argument("--restart-1", type=Path, required=True)
    parser.add_argument("--restart-2", type=Path, required=True)
    parser.add_argument("--final-restart", type=Path, required=True)
    parser.add_argument("--restart-from", type=Path, default=None)
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    if args.restart_from is not None and not args.restart_from.is_file():
        raise FileNotFoundError(f"Restart file not found: {args.restart_from}")

    atoms, cell = parse_pdb(args.pdb)

    metadata = write_lammps_data(
        path=args.data,
        atoms=atoms,
        cell=cell,
        element_order=args.element_order,
    )

    if args.restart_from is not None:
        warmup_steps = 0
        init_velocities = False
    else:
        warmup_steps = args.warmup_steps
        init_velocities = args.init_velocities

    write_lammps_input(
        path=args.input,
        data_path=args.data.resolve(),
        model_path=args.model.resolve(),
        pair_coeff_elements=list(metadata["pair_coeff_elements"]),
        temperature=args.temperature,
        timestep_fs=args.timestep_fs,
        friction_fs_inv=args.friction_fs_inv,
        warmup_steps=warmup_steps,
        steps=args.steps,
        thermo_interval=args.thermo_interval,
        dump_interval=args.dump_interval,
        xtc_dump_interval=args.xtc_dump_interval,
        xtc_path=args.xtc_path.resolve(),
        seed=args.seed,
        init_velocities=init_velocities,
        final_data_path=args.final_data.resolve(),
        dump_path=args.dump.resolve(),
        restart_interval=args.restart_interval,
        restart_path_1=args.restart_1.resolve(),
        restart_path_2=args.restart_2.resolve(),
        final_restart_path=args.final_restart.resolve(),
        restart_from=args.restart_from,
    )

    payload = {
        "pdb": str(args.pdb),
        "data": str(args.data),
        "input": str(args.input),
        "model": str(args.model),
        "temperature": args.temperature,
        "timestep_fs": args.timestep_fs,
        "friction_fs_inv": args.friction_fs_inv,
        "warmup_steps": warmup_steps,
        "steps": args.steps,
        "thermo_interval": args.thermo_interval,
        "dump_interval": args.dump_interval,
        "xtc_dump_interval": args.xtc_dump_interval,
        "xtc_path": str(args.xtc_path),
        "seed": args.seed,
        "init_velocities": init_velocities,
        "restart_interval": args.restart_interval,
        "restart_from": None if args.restart_from is None else str(args.restart_from.resolve()),
        "restart_1": str(args.restart_1),
        "restart_2": str(args.restart_2),
        "final_restart": str(args.final_restart),
        **metadata,
    }

    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"wrote {args.data}")
    print(f"wrote {args.input}")
    print(f"wrote {args.metadata}")
    print(f"restart_from={args.restart_from}")
    print(f"restart_interval={args.restart_interval}")
    print(f"final_restart={args.final_restart}")
    print(f"pair_coeff * * {' '.join(metadata['pair_coeff_elements'])}")

if __name__ == "__main__":
    main()
