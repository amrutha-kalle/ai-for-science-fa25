"""
Script to:
1. Extract chemical composition from CIF files
2. Run chemeleon with chemical composition as text input
3. Run chemeleon with CIF structure as text input
4. Compare outputs pairwise
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle
import numpy as np
from collections import defaultdict

import ase
from ase.io import read as ase_read
from ase.io import write as ase_write
import pymatgen
from pymatgen.core import Structure
from pymatgen.io.cif import CifParser

# Import chemeleon
from chemeleon import Chemeleon


def extract_composition_from_cif(cif_path: str) -> str:
    """
    Extract chemical composition from a CIF file.
    
    Args:
        cif_path: Path to the CIF file
        
    Returns:
        Chemical composition as a string (e.g., "CaO10")
    """
    try:
        # Try using pymatgen first
        parser = CifParser(cif_path)
        structure = parser.get_structures()[0]
        composition = structure.composition
        # Format: "Ca1 O10" -> "CaO10"
        comp_str = str(composition)
        # Remove spaces
        comp_str = comp_str.replace(" ", "")
        return comp_str
    except Exception as e:
        print(f"Error parsing {cif_path} with pymatgen: {e}")
        try:
            # Fallback to ASE
            atoms = ase_read(cif_path)
            # Get unique atoms and their counts
            symbols = atoms.get_chemical_symbols()
            symbol_counts = defaultdict(int)
            for symbol in symbols:
                symbol_counts[symbol] += 1
            
            # Sort by symbol for consistency
            comp_str = "".join(f"{k}{v}" for k, v in sorted(symbol_counts.items()))
            return comp_str
        except Exception as e2:
            print(f"Error parsing {cif_path} with ASE: {e2}")
            return None


def extract_cif_content(cif_path: str) -> Optional[str]:
    """
    Extract CIF file content as text (first 2000 characters for prompt).
    
    Args:
        cif_path: Path to the CIF file
        
    Returns:
        CIF content as string or None if failed
    """
    try:
        with open(cif_path, 'r') as f:
            content = f.read()
            return content[:2000]
    except Exception as e:
        print(f"Error reading CIF content from {cif_path}: {e}")
        return None


def load_chemeleon_model():
    """Load the chemeleon model."""
    print("Loading chemeleon model...")
    try:
        chemeleon = Chemeleon.load_general_text_model()
        print("✓ Model loaded successfully")
        return chemeleon
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def run_chemeleon_sample(
    chemeleon,
    text_input: str,
    n_atoms: int = 6,
    n_samples: int = 3
) -> Optional[List]:
    """
    Run chemeleon sampling with text input.
    
    Args:
        chemeleon: Chemeleon model instance
        text_input: Text prompt for structure generation
        n_atoms: Number of atoms in the unit cell
        n_samples: Number of samples to generate
        
    Returns:
        List of generated ASE Atoms objects or None if failed
    """
    try:
        atoms_list = chemeleon.sample(text_input, n_atoms, n_samples)
        return atoms_list
    except Exception as e:
        print(f"Error during sampling: {e}")
        return None


def extract_structure_info(atoms_obj) -> Dict:
    """
    Extract structural information from an ASE Atoms object.
    
    Args:
        atoms_obj: ASE Atoms object
        
    Returns:
        Dictionary with structural properties
    """
    info = {
        "n_atoms": len(atoms_obj),
        "cell": atoms_obj.cell.cellpar().tolist(),  # [a, b, c, alpha, beta, gamma]
        "composition": "".join(f"{k}{v}" for k, v in sorted(
            defaultdict(int, {s: atoms_obj.get_chemical_symbols().count(s) 
                            for s in set(atoms_obj.get_chemical_symbols())}).items()
        )),
        "volume": atoms_obj.get_volume(),
        "symbols": atoms_obj.get_chemical_symbols(),
    }
    return info


def calculate_similarity(info1: Dict, info2: Dict) -> Dict:
    """
    Calculate similarity between two structures.
    
    Args:
        info1: Structure info from first generation
        info2: Structure info from second generation
        
    Returns:
        Dictionary with similarity metrics
    """
    similarity = {
        "same_composition": info1["composition"] == info2["composition"],
        "n_atoms_diff": abs(info1["n_atoms"] - info2["n_atoms"]),
        "cell_similarity": None,
        "volume_diff": abs(info1["volume"] - info2["volume"]),
        "volume_diff_percent": abs(info1["volume"] - info2["volume"]) / max(info1["volume"], info2["volume"]) * 100,
    }
    
    # Calculate cell parameter similarity
    if info1["cell"] and info2["cell"]:
        cell_params_1 = np.array(info1["cell"])
        cell_params_2 = np.array(info2["cell"])
        # Normalize and compare
        cell_similarity = 1 - np.mean(np.abs(cell_params_1 - cell_params_2) / (cell_params_1 + 1e-6))
        similarity["cell_similarity"] = max(0, cell_similarity)
    
    return similarity


def process_cif_files(
    cif_dir: str,
    output_dir: str = "chemeleon_analysis_results",
    max_cifs: Optional[int] = None
) -> None:
    """
    Main processing function: Extract compositions, run chemeleon, and compare.
    
    Args:
        cif_dir: Directory containing extracted CIF files
        output_dir: Directory to save results
        max_cifs: Maximum number of CIF files to process (None = all)
    """
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Get list of CIF files
    cif_files = sorted(Path(cif_dir).glob("*.cif"))
    if max_cifs:
        cif_files = cif_files[:max_cifs]
    
    print(f"\nFound {len(cif_files)} CIF files")
    
    # Load chemeleon model
    chemeleon = load_chemeleon_model()
    if chemeleon is None:
        print("Failed to load chemeleon model. Exiting.")
        return
    
    results = {
        "total_cifs": len(cif_files),
        "successful_extractions": 0,
        "successful_generations": 0,
        "compositions": [],
        "comparisons": [],
    }
    
    # Process each CIF
    for idx, cif_path in enumerate(cif_files):
        print(f"\n[{idx+1}/{len(cif_files)}] Processing: {cif_path.name}")
        
        # Extract composition
        composition = extract_composition_from_cif(str(cif_path))
        if composition is None:
            print(f"✗ Failed to extract composition from {cif_path.name}")
            continue
        
        print(f"✓ Composition: {composition}")
        results["successful_extractions"] += 1
        
        # Prepare text prompts
        n_atoms = len(composition)  # Rough estimate
        n_atoms = min(max(n_atoms, 6), 40)  # Clamp between 6 and 40
        
        # Prompt 1: Chemical composition as text
        text_prompt_1 = f"A crystal structure of {composition}"
        
        # Prompt 2: CIF file content as text input
        cif_content = extract_cif_content(str(cif_path))
        if cif_content is None:
            print(f"✗ Failed to extract CIF content from {cif_path.name}")
            continue
        text_prompt_2 = cif_content
        
        print(f"Generating samples with n_atoms={n_atoms}...")
        print(f"  Prompt 1: Composition-based")
        print(f"  Prompt 2: CIF content-based ({len(cif_content)} chars)")
        
        # Run chemeleon with composition-based prompt
        samples_1 = run_chemeleon_sample(chemeleon, text_prompt_1, n_atoms, n_samples=2)
        # Run chemeleon with CIF content-based prompt
        samples_2 = run_chemeleon_sample(chemeleon, text_prompt_2, n_atoms, n_samples=2)
        
        if samples_1 is None or samples_2 is None:
            print(f"✗ Failed to generate samples for {composition}")
            continue
        
        print(f"✓ Generated {len(samples_1)} + {len(samples_2)} samples")
        results["successful_generations"] += 1
        
        # Extract structural info and compare
        info_samples_1 = [extract_structure_info(atoms) for atoms in samples_1]
        info_samples_2 = [extract_structure_info(atoms) for atoms in samples_2]
        
        # Save generated CIF structures
        cif_base_name = cif_path.stem  # Remove .cif extension
        gen_cifs_dir = Path(output_dir) / "generated_cifs" / cif_base_name
        gen_cifs_dir.mkdir(parents=True, exist_ok=True)
        
        sample_1_paths = []
        sample_2_paths = []
        
        # Save prompt 1 samples (composition-based)
        for i, atoms in enumerate(samples_1):
            cif_file = gen_cifs_dir / f"prompt1_sample{i}.cif"
            ase_write(str(cif_file), atoms)
            sample_1_paths.append(str(cif_file))
        
        # Save prompt 2 samples (CIF-content-based)
        for i, atoms in enumerate(samples_2):
            cif_file = gen_cifs_dir / f"prompt2_sample{i}.cif"
            ase_write(str(cif_file), atoms)
            sample_2_paths.append(str(cif_file))
        
        print(f"✓ Saved {len(samples_1)} + {len(samples_2)} CIF files to {gen_cifs_dir}")
        
        # Pairwise comparisons
        comparisons = {
            "cif_name": cif_path.name,
            "composition": composition,
            "text_prompt_1": text_prompt_1,
            "text_prompt_1_type": "composition-based",
            "text_prompt_2_type": "cif-content-based",
            "sample_1_paths": sample_1_paths,
            "sample_2_paths": sample_2_paths,
            "samples_1_info": info_samples_1,
            "samples_2_info": info_samples_2,
            "pairwise_similarities": [],
        }
        
        # Compare each sample from prompt 1 with each from prompt 2
        for i, info1 in enumerate(info_samples_1):
            for j, info2 in enumerate(info_samples_2):
                sim = calculate_similarity(info1, info2)
                sim["sample_1_idx"] = i
                sim["sample_2_idx"] = j
                comparisons["pairwise_similarities"].append(sim)
        
        results["compositions"].append({
            "cif_name": cif_path.name,
            "composition": composition,
        })
        results["comparisons"].append(comparisons)
        
        # Save intermediate results every 5 CIFs
        if (idx + 1) % 5 == 0:
            results_file = Path(output_dir) / "intermediate_results.json"
            with open(results_file, "w") as f:
                # Convert numpy arrays to lists for JSON serialization
                json_results = json.loads(json.dumps(results, default=str))
                json.dump(json_results, f, indent=2)
            print(f"✓ Saved intermediate results to {results_file}")
    
    # Save final results
    print(f"\n\n=== SUMMARY ===")
    print(f"Total CIFs processed: {len(cif_files)}")
    print(f"Successful extractions: {results['successful_extractions']}")
    print(f"Successful generations: {results['successful_generations']}")
    
    # Save results
    results_file = Path(output_dir) / "final_results.json"
    compositions_file = Path(output_dir) / "compositions.json"
    comparisons_file = Path(output_dir) / "comparisons.pkl"
    
    with open(results_file, "w") as f:
        json_results = json.loads(json.dumps(results, default=str))
        json.dump(json_results, f, indent=2)
    
    with open(compositions_file, "w") as f:
        comps = [c for c in results["compositions"]]
        json.dump(comps, f, indent=2)
    
    with open(comparisons_file, "wb") as f:
        pickle.dump(results["comparisons"], f)
    
    print(f"✓ Saved results to {output_dir}/")
    print(f"  - final_results.json")
    print(f"  - compositions.json")
    print(f"  - comparisons.pkl")
    print(f"")
    print(f"Analysis compared:")
    print(f"  Prompt 1: Composition-based (e.g., 'A crystal structure of K2U1Pd2S2')")
    print(f"  Prompt 2: CIF-content-based (using actual CIF file text as input)")
    
    # Generate summary statistics
    generate_summary_report(results, output_dir)


def generate_summary_report(results: Dict, output_dir: str) -> None:
    """
    Generate a summary report of the analysis.
    
    Args:
        results: Results dictionary from processing
        output_dir: Directory to save the report
    """
    report_lines = [
        "=" * 80,
        "CHEMELEON CIF ANALYSIS REPORT",
        "=" * 80,
        "",
        f"Total CIFs Processed: {results['total_cifs']}",
        f"Successful Extractions: {results['successful_extractions']}",
        f"Successful Generations: {results['successful_generations']}",
        "",
        "UNIQUE COMPOSITIONS EXTRACTED:",
        "-" * 80,
    ]
    
    # Get unique compositions
    unique_comps = {}
    for comp_info in results["compositions"]:
        comp = comp_info["composition"]
        if comp not in unique_comps:
            unique_comps[comp] = []
        unique_comps[comp].append(comp_info["cif_name"])
    
    for comp in sorted(unique_comps.keys()):
        count = len(unique_comps[comp])
        report_lines.append(f"  {comp}: {count} CIF(s)")
    
    report_lines.extend([
        "",
        "STRUCTURAL COMPARISON STATISTICS:",
        "-" * 80,
    ])
    
    # Calculate statistics from comparisons
    if results["comparisons"]:
        same_comp_count = 0
        total_comparisons = 0
        avg_volume_diff = []
        avg_cell_similarity = []
        
        for comp in results["comparisons"]:
            for sim in comp["pairwise_similarities"]:
                total_comparisons += 1
                if sim["same_composition"]:
                    same_comp_count += 1
                avg_volume_diff.append(sim["volume_diff_percent"])
                if sim["cell_similarity"] is not None:
                    avg_cell_similarity.append(sim["cell_similarity"])
        
        report_lines.extend([
            f"  Total Pairwise Comparisons: {total_comparisons}",
            f"  Comparisons with Same Composition: {same_comp_count} ({same_comp_count/total_comparisons*100:.1f}%)",
            f"  Average Volume Difference: {np.mean(avg_volume_diff):.2f}%",
            f"  Average Cell Similarity: {np.mean(avg_cell_similarity):.3f}" if avg_cell_similarity else "",
            "",
        ])
    
    report_lines.extend([
        "=" * 80,
    ])
    
    report_text = "\n".join(report_lines)
    
    report_file = Path(output_dir) / "REPORT.txt"
    with open(report_file, "w") as f:
        f.write(report_text)
    
    print("\n" + report_text)
    print(f"✓ Report saved to {report_file}")


if __name__ == "__main__":
    # Configuration
    CIF_DIR = "../data/extracted_cifs"
    OUTPUT_DIR = "../results/chemeleon_analysis_results"
    MAX_CIFS = None  # Set to a number to limit processing (e.g., 10 for testing)
    
    print("Starting CIF analysis with Chemeleon...")
    print(f"CIF Directory: {CIF_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    
    process_cif_files(CIF_DIR, OUTPUT_DIR, MAX_CIFS)
    
    print("\nAnalysis complete!")
