from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core import Composition, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer, SymmetryUndeterminedError
from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import argparse
import re
from collections import defaultdict
from pymatgen.core import Composition
from pymatgen.io.cif import CifBlock
from pymatgen.symmetry.groups import SpaceGroup
from pymatgen.core.operations import SymmOp
from ase.optimize import BFGS
from ase.vibrations import Vibrations
from mace.calculators import mace_mp
from ase import Atoms
import numpy as np
import os
import csv
import sys


# ----------------------------
# Convert CIF string → ASE Atoms
# ----------------------------
def cif_to_ase(cif_string):
    struct = Structure.from_str(cif_string, fmt="cif")
    symbols = [str(s.specie) for s in struct]
    positions = struct.frac_coords @ struct.lattice.matrix
    cell = struct.lattice.matrix
    return Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)

# ----------------------------
# Relax structure using MLFF
# ----------------------------
def relax_structure(atoms):
    atoms = atoms.copy()
    calc = mace_mp()
    atoms.set_calculator(calc)
    dyn = BFGS(atoms, logfile=None)
    dyn.run(fmax=0.03)     # convergence threshold
    return atoms

# ----------------------------
# Compute phonons (local minimum check)
# ----------------------------
def compute_phonons(atoms, indices=None):
    """
    indices = list of atom indices to displace, or None for all atoms.
    """
    atoms = atoms.copy()
    calc = mace_mp()
    atoms.set_calculator(calc)

    vib = Vibrations(atoms, indices=indices)
    vib.run()
    freqs = vib.get_frequencies()  # in cm^-1
    vib.clean()
    return freqs

# ----------------------------
# Full metastability evaluation
# ----------------------------
def evaluate_metastability(cif_string, phonon_subset=None):
    atoms = cif_to_ase(cif_string)

    # 1. Relaxation
    relaxed = relax_structure(atoms)
    energy = relaxed.get_potential_energy() / len(relaxed)

    # 2. Dynamical stability via phonons
    freqs = compute_phonons(relaxed, indices=phonon_subset)
    n_imag = np.sum(freqs < 0)

    return {
        "energy_eV_per_atom": energy,
        "num_imaginary_modes": int(n_imag),
        "is_dynamically_stable": (n_imag == 0)
    }

def extract_data_formula(cif_str):
    """
    Try to grab whatever comes after 'data_' on the first data line.
    This might NOT be a real formula (e.g. 'image0'), so callers must
    treat it as optional / best-effort.
    """
    match = re.search(r"^data_([^\n]+)", cif_str, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return None


# Returns true if chemical formula is consistent throughout the generated CIF
def formula_consistent(cif_str):
    try:
        parser = CifParser.from_str(cif_str)
        cif_data = parser.as_dict()
        block = cif_data[list(cif_data.keys())[0]]
    except Exception:
        # If we can't parse, we definitely can't say it's consistent
        return False

    formula_sum_str = block.get("_chemical_formula_sum", None)
    formula_struct_str = block.get("_chemical_formula_structural", None)

    if formula_sum_str is None or formula_struct_str is None:
        # Missing info → mark as not consistent (so bad CIFs get penalized)
        return False

    try:
        formula_sum = Composition(formula_sum_str)
        formula_structural = Composition(formula_struct_str)
    except Exception:
        return False

    # Optional: try to use data_ label if it's a valid formula
    data_formula_str = extract_data_formula(cif_str)
    if data_formula_str:
        try:
            formula_data = Composition(data_formula_str)
            return (
                formula_data.reduced_formula
                == formula_sum.reduced_formula
                == formula_structural.reduced_formula
            )
        except Exception:
            # Ignore bogus data_ labels like "image0"
            pass

    return formula_sum.reduced_formula == formula_structural.reduced_formula

# Returns true if the atom site multiplicty is consistent throughout the generated CIF
def atom_site_multiplicity_consistent(cif_str, tol=1e-3):
    """
    Check that the total atom counts implied by the site multiplicities
    match the formula sum, up to a small tolerance.
    """
    try:
        parser = CifParser.from_str(cif_str)
        cif_data = parser.as_dict()
        block = cif_data[list(cif_data.keys())[0]]
    except Exception:
        return False

    formula_sum_str = block.get("_chemical_formula_sum", None)
    if formula_sum_str is None:
        return False

    try:
        expected_comp = Composition(formula_sum_str).as_dict()
    except Exception:
        return False

    expected_atoms = {str(el): float(count) for el, count in expected_comp.items()}
    actual_atoms = defaultdict(float)

    for key, subblock in cif_data.items():
        if not isinstance(subblock, dict):
            continue
        if "_atom_site_type_symbol" in subblock and "_atom_site_symmetry_multiplicity" in subblock:
            types = subblock["_atom_site_type_symbol"]
            mults = subblock["_atom_site_symmetry_multiplicity"]
            occs = subblock.get("_atom_site_occupancy", [1.0] * len(types))

            for atom_type, multiplicity, occ in zip(types, mults, occs):
                try:
                    m = float(multiplicity)
                except Exception:
                    continue
                try:
                    o = float(occ)
                except Exception:
                    o = 1.0

                actual_atoms[atom_type] += m * o

    for el, exp_count in expected_atoms.items():
        act_count = actual_atoms.get(el, 0.0)
        if abs(act_count - exp_count) > tol:
            return False

    for el, act_count in actual_atoms.items():
        if el not in expected_atoms and abs(act_count) > tol:
            return False

    return True


# if the imports above complain, you can just catch a generic Exception instead


import re
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

def _get_stated_space_group(block):
    """
    Best-effort lookup for the stated H-M space group symbol
    from a CIF block dict.
    """
    candidate_keys = [
        "_symmetry_space_group_name_H-M",
        "_space_group_name_H-M_alt",
        "_space_group_name_H-M",
        "_symmetry_space_group_name_H-M_alt",
    ]
    for key in candidate_keys:
        if key in block:
            val = block[key]
            # strip quotes and whitespace
            s = str(val).strip().strip('"').strip("'")
            return s
    return None

def _normalize_sg_symbol(symbol):
    if symbol is None:
        return None
    s = str(symbol)
    # strip surrounding quotes and whitespace
    s = s.strip().strip('"').strip("'")
    # remove internal spaces (P 1 -> P1, P -1 -> P-1)
    s = re.sub(r"\s+", "", s)
    # upper-case for safety
    return s.upper()

def space_group_consistent(cif_str):
    """
    Returns True if stated space group (from CIF tags) matches
    the detected one (from spglib), after normalization.
    Returns False on parse/symmetry failures or obvious mismatch.
    """
    try:
        structure = Structure.from_str(cif_str, fmt="cif")
    except Exception:
        # Can't even get a structure → inconsistent
        return False

    try:
        parser = CifParser.from_str(cif_str)
        cif_data = parser.as_dict()
        block = cif_data[list(cif_data.keys())[0]]
    except Exception:
        return False

    stated_raw = _get_stated_space_group(block)
    if stated_raw is None:
        # No stated SG → can't check → mark inconsistent
        return False

    try:
        sga = SpacegroupAnalyzer(structure, symprec=0.1)
        detected_raw = sga.get_space_group_symbol()
    except Exception:
        return False

    stated = _normalize_sg_symbol(stated_raw)
    detected = _normalize_sg_symbol(detected_raw)

    return stated == detected


# Determine if the bond length is reasonable
def bond_length_reasonableness_score(cif_str, tolerance=0.32, h_factor=2.5):
    """
    Returns a normalized score [0,1]-ish; 0 if structure cannot be parsed
    or if CrystalNN fails.
    """
    try:
        structure = Structure.from_str(cif_str, fmt="cif")
    except Exception:
        return 0.0

    crystal_nn = CrystalNN()
    min_ratio = 1 - tolerance
    max_ratio = 1 + tolerance

    score = 0
    bond_count = 0

    try:
        for i, site in enumerate(structure):
            bonded_sites = crystal_nn.get_nn_info(structure, i)
            for connected_site_info in bonded_sites:
                j = connected_site_info["site_index"]
                if i == j:
                    continue
                connected_site = connected_site_info["site"]
                bond_length = site.distance(connected_site)

                is_hydrogen_bond = "H" in [site.specie.symbol, connected_site.specie.symbol]

                # handle missing X gracefully
                if site.specie.X is None or connected_site.specie.X is None:
                    electronegativity_diff = 0.0
                else:
                    electronegativity_diff = abs(site.specie.X - connected_site.specie.X)

                if electronegativity_diff >= 1.7:
                    try:
                        if site.specie.X < connected_site.specie.X:
                            expected_length = (
                                site.specie.average_cationic_radius
                                + connected_site.specie.average_anionic_radius
                            )
                        else:
                            expected_length = (
                                site.specie.average_anionic_radius
                                + connected_site.specie.average_cationic_radius
                            )
                    except Exception:
                        expected_length = site.specie.atomic_radius + connected_site.specie.atomic_radius
                else:
                    expected_length = site.specie.atomic_radius + connected_site.specie.atomic_radius

                if not expected_length:
                    continue

                bond_ratio = bond_length / expected_length

                if is_hydrogen_bond:
                    if bond_ratio < h_factor:
                        score += 1
                else:
                    if min_ratio < bond_ratio < max_ratio:
                        score += 1

                bond_count += 1
    except Exception:
        # Any weirdness in CrystalNN -> treat as bad
        return 0.0

    if bond_count == 0:
        return 0.0

    return score / bond_count


def is_valid(cif_str, bond_length_acceptability_cutoff=1.0):
    if not formula_consistent(cif_str):
        return False
    if not atom_site_multiplicity_consistent(cif_str):
        return False
    bond_length_score = bond_length_reasonableness_score(cif_str)
    if bond_length_score < bond_length_acceptability_cutoff:
        return False
    if not space_group_consistent(cif_str):
        return False
    return True

def is_sensible(cif_str, length_lo=0.5, length_hi=1000., angle_lo=10., angle_hi=170.):
    cell_length_pattern = re.compile(r"_cell_length_[abc]\s+([\d\.]+)")
    cell_angle_pattern = re.compile(r"_cell_angle_(alpha|beta|gamma)\s+([\d\.]+)")

    cell_lengths = cell_length_pattern.findall(cif_str)
    for length_str in cell_lengths:
        length = float(length_str)
        if length < length_lo or length > length_hi:
            return False

    cell_angles = cell_angle_pattern.findall(cif_str)
    for _, angle_str in cell_angles:
        angle = float(angle_str)
        if angle < angle_lo or angle > angle_hi:
            return False

    return True

def replace_symmetry_operators(cif_str, space_group_symbol):
    space_group = SpaceGroup(space_group_symbol)
    symmetry_ops = space_group.symmetry_ops

    loops = []
    data = {}
    symmops = []
    for op in symmetry_ops:
        v = op.translation_vector
        symmops.append(SymmOp.from_rotation_and_translation(op.rotation_matrix, v))

    ops = [op.as_xyz_string() for op in symmops]
    data["_symmetry_equiv_pos_site_id"] = [f"{i}" for i in range(1, len(ops) + 1)]
    data["_symmetry_equiv_pos_as_xyz"] = ops

    loops.append(["_symmetry_equiv_pos_site_id", "_symmetry_equiv_pos_as_xyz"])

    symm_block = str(CifBlock(data, loops, "")).replace("data_\n", "")

    pattern = r"(loop_\n_symmetry_equiv_pos_site_id\n_symmetry_equiv_pos_as_xyz\n1 'x, y, z')"
    cif_str_updated = re.sub(pattern, symm_block, cif_str)

    return cif_str_updated

def extract_space_group_symbol(cif_str):
    """
    Try to extract an H-M space group symbol from several possible tags.
    Returns a string like "P 1" or None if not found.
    """

    # 1) Standard CIF tag
    m = re.search(r"_symmetry_space_group_name_H-M\s+('([^']+)'|\"([^\"]+)\"|(\S+))", cif_str)
    if m:
        for g in m.groups()[1:]:
            if g:
                return g

    # 2) Alternative tag used in many generated / Materials Project CIFs
    m = re.search(r"_space_group_name_H-M_alt\s+('([^']+)'|\"([^\"]+)\"|(\S+))", cif_str)
    if m:
        for g in m.groups()[1:]:
            if g:
                return g

    # 3) Fallback: if IT number is 1, that's P 1
    m = re.search(r"_space_group_IT_number\s+(\d+)", cif_str)
    if m and m.group(1) == "1":
        return "P 1"

    # Nothing found → just say None instead of blowing up
    return None



def electronegativity_stats(cif_string):
    """
    Compute basic electronegativity stats from a CIF.
    If structure parsing fails or no EN data is available,
    return all fields as None but keep the same keys.
    """
    try:
        s = Structure.from_str(cif_string, fmt="cif")
    except Exception:
        # Cannot parse structure at all
        return {
            "mean_en": None,
            "std_en": None,
            "max_en": None,
            "min_en": None,
            "max_diff_pair": None,
            "per_site": [],
        }

    # Collect EN values, ignoring sites without X defined
    en_values = [site.specie.X for site in s if site.specie.X is not None]
    elements = [site.specie.symbol for site in s]

    if not en_values:
        # Parsed a structure but no EN data available
        return {
            "mean_en": None,
            "std_en": None,
            "max_en": None,
            "min_en": None,
            "max_diff_pair": None,
            "per_site": list(zip(elements, [None] * len(elements))),
        }

    en_arr = np.array(en_values)
    return {
        "mean_en": float(en_arr.mean()),
        "std_en": float(en_arr.std()),
        "max_en": float(en_arr.max()),
        "min_en": float(en_arr.min()),
        "max_diff_pair": float(en_arr.max() - en_arr.min()),
        "per_site": list(zip(elements, en_arr.tolist())),
    }



# given CIF as a string, figure out if it's valid
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        usage="given directory of cif files, measure validity and store results in csv"
    )
    parser.add_argument(
        "-d", "--dir", dest="cif_dir",
        help="path to directory containing cif files",
        required=True,
    )
    parser.add_argument(
        "-o", "--output", dest="csv_file",
        help="path to output csv file",
        default="./validity_results.csv",
    )
    parser.add_argument(
        "--skip-meta", dest="skip_meta",
        action="store_true",
        help="skip metastability calculation",
    )

    args = parser.parse_args()
    cif_dir = args.cif_dir
    csv_file = args.csv_file
    skip_meta = args.skip_meta

    rows = []

    # Recursively walk cif_dir and process every .cif file
    for root, _, files in os.walk(cif_dir):
        for filename in files:
            if not filename.lower().endswith(".cif"):
                continue

            path = os.path.join(root, filename)
            with open(path, "r") as f:
                cif = f.read()

            # Sensible?
            sens = is_sensible(cif)

            # Fix sym ops
            sg = extract_space_group_symbol(cif)
            if sg and sg != "P 1":
                cif = replace_symmetry_operators(cif, sg)

            # Compute validity components
            f_cons = formula_consistent(cif)
            a_cons = atom_site_multiplicity_consistent(cif)
            sg_cons = space_group_consistent(cif)
            bond = bond_length_reasonableness_score(cif)
            valid = f_cons and a_cons and sg_cons and bond >= 1.0

            # Metastability
            if skip_meta:
                meta = {
                    "energy_eV_per_atom": None,
                    "num_imaginary_modes": None,
                    "is_dynamically_stable": None,
                }
            else:
                meta = evaluate_metastability(cif)

            electro_stats = electronegativity_stats(cif)

            rows.append({
                # relative path so you can tell which mp-id it came from
                "filename": os.path.relpath(path, cif_dir),
                "is_sensible": sens,
                "formula_consistent": f_cons,
                "atom_site_consistent": a_cons,
                "space_group_consistent": sg_cons,
                "bond_length_score": bond,
                "is_valid": valid,
                "energy_eV_per_atom": meta["energy_eV_per_atom"],
                "num_imaginary_modes": meta["num_imaginary_modes"],
                "is_dynamically_stable": meta["is_dynamically_stable"],
                "mean_en": electro_stats["mean_en"],
                "std_en": electro_stats["std_en"],
                "max_en": electro_stats["max_en"],
                "min_en": electro_stats["min_en"],
                "max_diff_pair": electro_stats["max_diff_pair"],
                "per_site": electro_stats["per_site"],
            })

    # --- CSV writing with safety guard *after* the walk ---
    if not rows:
        print(f"No .cif files found under directory: {cif_dir}")
        sys.exit(1)

    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)



    # with open(cif_file, "r") as f:
    #     cif = f.read()
    # if not is_sensible(cif):
    #     print("CIF is not sensible")
    #     exit
    # space_group_symbol = extract_space_group_symbol(cif)
    # if space_group_symbol is not None and space_group_symbol != "P 1":
    #     cif = replace_symmetry_operators(cif, space_group_symbol)
    
    # if is_valid(cif):
    #     print("Provided CIF is valid")
    # else:
    #     print("Provided CIF is not valid")

    
    
