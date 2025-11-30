from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core import Composition, Structure
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
    match = re.search(r"data_([A-Za-z0-9]+)\n", cif_str)
    if match:
        return match.group(1)
    raise Exception(f"could not find data_ in:\n{cif_str}")

# Returns true if chemical formula is consistent throughout the generated CIF
def formula_consistent(cif_str):
    parser = CifParser.from_string(cif_str)
    cif_data = parser.as_dict()

    formula_data = Composition(extract_data_formula(cif_str))
    formula_sum = Composition(cif_data[list(cif_data.keys())[0]]["_chemical_formula_sum"])
    formula_structural = Composition(cif_data[list(cif_data.keys())[0]]["_chemical_formula_structural"])

    return formula_data.reduced_formula == formula_sum.reduced_formula == formula_structural.reduced_formula

# Returns true if the atom site multiplicty is consistent throughout the generated CIF
def atom_site_multiplicity_consistent(cif_str):
    parser = CifParser.from_string(cif_str)
    cif_data = parser.as_dict()

    formula_sum = cif_data[list(cif_data.keys())[0]]["_chemical_formula_sum"]
    expected_atoms = Composition(formula_sum).as_dict()

    actual_atoms = defaultdict(int)
    for key in cif_data:
        if "_atom_site_type_symbol" in cif_data[key] and "_atom_site_symmetry_multiplicity" in cif_data[key]:
            for atom_type, multiplicity in zip(cif_data[key]["_atom_site_type_symbol"],
                                               cif_data[key]["_atom_site_symmetry_multiplicity"]):
                actual_atoms[atom_type] += int(multiplicity)

    return expected_atoms == actual_atoms

# Returns true if the stated space group is consistent with the detected space group
def space_group_consistent(cif_str):
    structure = Structure.from_str(cif_str, fmt="cif")
    parser = CifParser.from_string(cif_str)
    cif_data = parser.as_dict()

    # Extract the stated space group from the CIF file
    stated_space_group = cif_data[list(cif_data.keys())[0]]['_symmetry_space_group_name_H-M']

    # Analyze the symmetry of the structure
    spacegroup_analyzer = SpacegroupAnalyzer(structure, symprec=0.1)

    # Get the detected space group
    detected_space_group = spacegroup_analyzer.get_space_group_symbol()

    # Check if the detected space group matches the stated space group
    is_match = stated_space_group.strip() == detected_space_group.strip()

    return is_match

# Determine if the bond length is reasonable
def bond_length_reasonableness_score(cif_str, tolerance=0.32, h_factor=2.5):
    """
    If a bond length is 30% shorter or longer than the sum of the atomic radii, the score is lower.
    """
    structure = Structure.from_str(cif_str, fmt="cif")
    crystal_nn = CrystalNN()

    min_ratio = 1 - tolerance
    max_ratio = 1 + tolerance

    # calculate the score based on bond lengths and covalent radii
    score = 0
    bond_count = 0
    for i, site in enumerate(structure):
        bonded_sites = crystal_nn.get_nn_info(structure, i)
        for connected_site_info in bonded_sites:
            j = connected_site_info['site_index']
            if i == j:  # skip if they're the same site
                continue
            connected_site = connected_site_info['site']
            bond_length = site.distance(connected_site)

            is_hydrogen_bond = "H" in [site.specie.symbol, connected_site.specie.symbol]

            electronegativity_diff = abs(site.specie.X - connected_site.specie.X)
            """
            According to the Pauling scale, when the electronegativity difference 
            between two bonded atoms is less than 1.7, the bond can be considered 
            to have predominantly covalent character, while a difference greater 
            than or equal to 1.7 indicates that the bond has significant ionic 
            character.
            """
            if electronegativity_diff >= 1.7:
                # use ionic radii
                if site.specie.X < connected_site.specie.X:
                    expected_length = site.specie.average_cationic_radius + connected_site.specie.average_anionic_radius
                else:
                    expected_length = site.specie.average_anionic_radius + connected_site.specie.average_cationic_radius
            else:
                expected_length = site.specie.atomic_radius + connected_site.specie.atomic_radius

            bond_ratio = bond_length / expected_length

            # penalize bond lengths that are too short or too long;
            #  check if bond involves hydrogen and adjust tolerance accordingly
            if is_hydrogen_bond:
                if bond_ratio < h_factor:
                    score += 1
            else:
                if min_ratio < bond_ratio < max_ratio:
                    score += 1

            bond_count += 1

    normalized_score = score / bond_count

    return normalized_score

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
    match = re.search(r"_symmetry_space_group_name_H-M\s+('([^']+)'|(\S+))", cif_str)
    if match:
        return match.group(2) if match.group(2) else match.group(3)
    raise Exception(f"could not extract space group from:\n{cif_str}")


# given CIF as a string, figure out if it's valid
if __name__ == "__main__":
    parser = argparse.ArgumentParser(usage="given directory of cif files, measure validity and store results in cif")
    parser.add_argument("-d", "--dir", dest="cif_dir", help="path to directory containing cif files", required=True)
    parser.add_argument("-o", "--output", dest="csv_file", help="path to output csv file", default="./validity_results.csv")
    parser.add_argument("--skip-meta", dest="skip_meta", action="store_true", help="skip metastability calculation")

    args = parser.parse_args()
    cif_dir = args.cif_dir
    csv_file = args.csv_file
    skip_meta = args.skip_meta
    

    rows = []

    for filename in os.listdir(cif_dir):
        if not filename.lower().endswith(".cif"):
            continue

        path = os.path.join(cif_dir, filename)
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
                "is_dynamically_stable": None
            }
        else:
            meta = evaluate_metastability(cif)

        rows.append({
            "filename": filename,
            "is_sensible": sens,
            "formula_consistent": f_cons,
            "atom_site_consistent": a_cons,
            "space_group_consistent": sg_cons,
            "bond_length_score": bond,
            "is_valid": valid,
            "energy_eV_per_atom": meta["energy_eV_per_atom"],
            "num_imaginary_modes": meta["num_imaginary_modes"],
            "is_dynamically_stable": meta["is_dynamically_stable"],
        })

    # Write CSV
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

    
    

