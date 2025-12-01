"""
Script to:
1. Read CIF file paths from a CSV file
2. Extract chemical composition from CIF files
3. Run chemeleon with chemical composition as text input
4. Run chemeleon with CIF structure as text input
5. Compare outputs pairwise
"""

import os
import json
import re
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle
import numpy as np
from collections import defaultdict
import torch
from chemeleon.text_encoder.crystal_clip import CrystalClip


import ase
from ase.io import read as ase_read
from ase.io import write as ase_write
import pymatgen
from pymatgen.core import Structure
from pymatgen.io.cif import CifParser

# Import chemeleon
from chemeleon import Chemeleon


def load_cif_data_from_csv(csv_path: str) -> List[Dict]:
    """
    Load CIF structures and metadata from a CSV file.
    The CSV's 'cif' column contains the actual CIF structure content (not file paths).
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        List of dicts with 'cif_content' and 'material_id' keys
    """
    cif_data = []
    try:
        with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                if not row or 'cif' not in row:
                    continue
                
                cif_content = row['cif'].strip()
                if not cif_content or len(cif_content) < 50:  # Skip very short/empty entries
                    continue
                
                material_id = row.get('material_id', f"cif_{len(cif_data)}")
                cif_data.append({
                    'cif_content': cif_content,
                    'material_id': material_id,
                    'composition': row.get('composition', ''),
                })
    
    except Exception as e:
        print(f"Error reading CSV file {csv_path}: {e}")
        return []
    
    print(f"✓ Loaded {len(cif_data)} CIF structures from CSV")
    return cif_data


def extract_composition_from_cif(cif_path: str) -> Optional[str]:
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
    Extract CIF file content as text.
    
    Args:
        cif_path: Path to the CIF file
        
    Returns:
        CIF content as string or None if failed
    """
    try:
        with open(cif_path, 'r') as f:
            content = f.read()
            return content
    except Exception as e:
        print(f"Error reading CIF content from {cif_path}: {e}")
        return None


def load_chemeleon_model(checkpoint_path: Optional[str] = None, clip_checkpoint_path: Optional[str] = None):
    """
    Load the chemeleon model from checkpoints or default model.
    
    Args:
        checkpoint_path: Path to a custom Chemeleon checkpoint.
                        If None, looks for local checkpoint in chemeleon/checkpoints/
        clip_checkpoint_path: Path to a custom CLIP checkpoint.
                             If None, looks for local checkpoint in chemeleon/checkpoints/
        
    Returns:
        Chemeleon model instance or None if failed
    """
    print("Loading chemeleon model...")
    try:
        import os
        
        # # Use provided paths or defaults
        # if checkpoint_path is None:
        #     checkpoint_path = "/Users/sathv/ai-for-science-fa25/chemeleon/chemeleon/checkpoints/chemeleon-7fsg68c3.ckpt"
        # if clip_checkpoint_path is None:
        #     clip_checkpoint_path = "/Users/sathv/ai-for-science-fa25/chemeleon/chemeleon/checkpoints/clip-upy53q4b.ckpt"
        
        checkpoint_path = os.path.abspath(checkpoint_path)
        clip_checkpoint_path = os.path.abspath(clip_checkpoint_path)
        
        print(f"Chemeleon: {checkpoint_path}")
        print(f"CLIP: {clip_checkpoint_path}")
        
        try:
            # Load CLIP checkpoint with proper GEP initialization
            # We need to: 1) load checkpoint to get hparams, 2) create model with hparams, 3) load state dict
            print("Loading CLIP checkpoint...")
            clip = CrystalClip.load_from_checkpoint(
                clip_checkpoint_path,
                map_location="cuda",
            )
            # Load the trained weights
            clip.eval()
            print(f"CLIP model loaded successfully (with GEP modules)")
            
            # Load Chemeleon checkpoint
            print("Loading Chemeleon checkpoint...")
            chemeleon = Chemeleon.load_from_checkpoint(
                checkpoint_path,
                path_ckpt_clip=clip_checkpoint_path,
                map_location="cuda",
            )
            chemeleon.eval()
            print(f"Chemeleon model loaded successfully")
        finally:
            # Always restore original directory and remove patch
            # os.chdir(original_cwd)
            # CrystalClip.load_from_checkpoint = original_load
            pass
        
        print("Full Model loaded successfully")
        return chemeleon
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
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
    cif_csv_path: str,
    output_dir: str = "chemeleon_analysis_results",
    max_cifs: Optional[int] = None,
    checkpoint_path: Optional[str] = None,
    comparison_modes: Optional[List[str]] = None,
    clip_checkpoint_path: Optional[str] = None,
    cif_base_dir: Optional[str] = None
) -> None:
    """
    Main processing function: Read CIFs from CSV, extract compositions, run chemeleon, and compare.
    
    Args:
        cif_csv_path: Path to CSV file containing CIF file paths
        output_dir: Directory to save results
        max_cifs: Maximum number of CIF files to process (None = all)
        checkpoint_path: Path to a custom Chemeleon checkpoint (None = use default model)
        comparison_modes: List of comparison modes. Options: 
                         "text" (composition), "cif" (CIF content), "text+cif" (both)
                         Default: ["text", "cif"] for backward compatibility
        clip_checkpoint_path: Path to a custom CLIP checkpoint (None = use default model)
        cif_base_dir: Optional base directory for relative CIF paths in CSV
    """
    if comparison_modes is None:
        comparison_modes = ["text", "cif"]
    
    valid_modes = {"text", "cif", "text+cif"}
    for mode in comparison_modes:
        if mode not in valid_modes:
            print(f"Invalid comparison mode: {mode}. Valid modes: {valid_modes}")
            return
    
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    
    cif_data = load_cif_data_from_csv(cif_csv_path)
    if not cif_data:
        print("No CIF structures loaded from CSV. Exiting.")
        return
    
    if max_cifs:
        cif_data = cif_data[:max_cifs]
    
    print(f"\nProcessing {len(cif_data)} CIF structures")
    print(f"Comparison modes: {', '.join(comparison_modes)}")
    
    # Load chemeleon model
    chemeleon = load_chemeleon_model(checkpoint_path, clip_checkpoint_path)
    if chemeleon is None:
        print("Failed to load chemeleon model. Exiting.")
        return
    
    results = {
        "total_cifs": len(cif_data),
        "successful_extractions": 0,
        "successful_generations": 0,
        "comparison_modes": comparison_modes,
        "compositions": [],
        "comparisons": [],
    }
    
    # Process each CIF structure
    for idx, cif_entry in enumerate(cif_data):
        print(f"\n[{idx+1}/{len(cif_data)}] Processing: {cif_entry['material_id']}")
        
        # Get or extract composition
        composition = cif_entry.get('composition', '')
        if not composition or composition.strip() == '':
            # Try to extract from CIF content
            try:
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.cif', delete=False) as tmp:
                    tmp.write(cif_entry['cif_content'])
                    tmp_path = tmp.name
                composition = extract_composition_from_cif(tmp_path)
                os.unlink(tmp_path)
            except:
                print(f"Failed to extract composition")
                continue
        
        if composition is None:
            print(f"Failed to extract composition")
            continue
        
        print(f"Composition: {composition}")
        results["successful_extractions"] += 1
        
        # Prepare text prompts
        n_atoms = len(composition)  # Rough estimate
        n_atoms = min(max(n_atoms, 6), 40)  # Clamp between 6 and 40
        
        # Get CIF content (first 2000 chars)
        cif_content = cif_entry['cif_content'][:2000]
        
        # Build prompts dictionary based on comparison modes
        prompts = {}
        if "text" in comparison_modes:
            prompts["text"] = f"A crystal structure of {composition}"
        if "cif" in comparison_modes:
            prompts["cif"] = cif_content
        if "text+cif" in comparison_modes:
            prompts["text+cif"] = f"A crystal structure of {composition}\n\n{cif_content}"
        
        print(f"Generating samples with n_atoms={n_atoms}...")
        for mode in prompts:
            print(f"  {mode}: {len(prompts[mode])} chars")
        
        # Run chemeleon for each prompt
        samples = {}
        for mode, prompt_text in prompts.items():
            samples[mode] = run_chemeleon_sample(chemeleon, prompt_text, n_atoms, n_samples=2)
            if samples[mode] is None:
                print(f"Failed to generate samples for {composition} with prompt '{mode}'")
                # Remove this mode from samples if generation failed
                del samples[mode]
        
        if not samples:
            print(f"Failed to generate samples for {composition} with any prompt")
            continue
        
        print(f"Generated samples for {len(samples)} prompt mode(s)")
        results["successful_generations"] += 1
        
        # Extract structural info for all samples
        info_samples = {}
        for mode, atoms_list in samples.items():
            info_samples[mode] = [extract_structure_info(atoms) for atoms in atoms_list]
        
        # Save generated CIF structures
        cif_base_name = cif_entry['material_id']
        gen_cifs_dir = output_path / "generated_cifs" / cif_base_name
        gen_cifs_dir.mkdir(parents=True, exist_ok=True)
        
        sample_paths = {}
        for mode, atoms_list in samples.items():
            sample_paths[mode] = []
            for i, atoms in enumerate(atoms_list):
                cif_file = gen_cifs_dir / f"{mode}_sample{i}.cif"
                ase_write(str(cif_file), atoms)
                sample_paths[mode].append(str(cif_file))
        
        print(f"Saved {sum(len(p) for p in sample_paths.values())} CIF files to {gen_cifs_dir}")
        
        # Pairwise comparisons
        comparisons = {
            "cif_name": cif_entry['material_id'],
            "composition": composition,
            "prompts": prompts,
            "sample_paths": sample_paths,
            "info_samples": info_samples,
            "pairwise_similarities": [],
        }
        
        # Compare all combinations of modes
        mode_list = list(samples.keys())
        for i, mode1 in enumerate(mode_list):
            for mode2 in mode_list[i+1:]:
                # Compare each sample from mode1 with each from mode2
                for idx1, info1 in enumerate(info_samples[mode1]):
                    for idx2, info2 in enumerate(info_samples[mode2]):
                        sim = calculate_similarity(info1, info2)
                        sim["mode_1"] = mode1
                        sim["mode_2"] = mode2
                        sim["sample_1_idx"] = idx1
                        sim["sample_2_idx"] = idx2
                        comparisons["pairwise_similarities"].append(sim)
        
        results["compositions"].append({
            "cif_name": cif_entry['material_id'],
            "composition": composition,
        })
        results["comparisons"].append(comparisons)
        
        # Save intermediate results every 5 CIFs
        if (idx + 1) % 5 == 0:
            results_file = output_path / "intermediate_results.json"
            with open(results_file, "w") as f:
                json_results = json.loads(json.dumps(results, default=str))
                json.dump(json_results, f, indent=2)
            print(f"✓ Saved intermediate results to {results_file}")
    
    # Save final results
    print(f"\n\n=== SUMMARY ===")
    print(f"Total CIFs processed: {len(cif_data)}")
    print(f"Successful extractions: {results['successful_extractions']}")
    print(f"Successful generations: {results['successful_generations']}")
    print(f"Comparison modes: {', '.join(comparison_modes)}")
    
    # Save results
    results_file = output_path / "final_results.json"
    compositions_file = output_path / "compositions.json"
    comparisons_file = output_path / "comparisons.pkl"
    
    with open(results_file, "w") as f:
        json_results = json.loads(json.dumps(results, default=str))
        json.dump(json_results, f, indent=2)
    
    with open(compositions_file, "w") as f:
        comps = [c for c in results["compositions"]]
        json.dump(comps, f, indent=2)
    
    with open(comparisons_file, "wb") as f:
        pickle.dump(results["comparisons"], f)
    
    print(f"Saved results to {output_dir}/")
    print(f"  - final_results.json")
    print(f"  - compositions.json")
    print(f"  - comparisons.pkl")
    
    # Generate summary statistics
    generate_summary_report(results, output_dir)


def generate_summary_report(results: Dict, output_dir: str) -> None:
    """
    Generate a summary report of the analysis.
    
    Args:
        results: Results dictionary from processing
        output_dir: Directory to save the report
    """
    comparison_modes = results.get("comparison_modes", ["text", "cif"])
    
    report_lines = [
        "=" * 80,
        "CHEMELEON CIF ANALYSIS REPORT",
        "=" * 80,
        "",
        f"Total CIFs Processed: {results['total_cifs']}",
        f"Successful Extractions: {results['successful_extractions']}",
        f"Successful Generations: {results['successful_generations']}",
        f"Comparison Modes: {', '.join(comparison_modes)}",
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
    
    # Calculate overall statistics
    if results["comparisons"]:
        same_comp_count = 0
        total_comparisons = 0
        avg_volume_diff = []
        avg_cell_similarity = []
        mode_pair_stats = defaultdict(lambda: {
            "count": 0,
            "same_comp_count": 0,
            "volume_diffs": [],
            "cell_similarities": []
        })
        
        for comp in results["comparisons"]:
            for sim in comp["pairwise_similarities"]:
                total_comparisons += 1
                if sim["same_composition"]:
                    same_comp_count += 1
                avg_volume_diff.append(sim["volume_diff_percent"])
                if sim["cell_similarity"] is not None:
                    avg_cell_similarity.append(sim["cell_similarity"])
                
                # Track statistics by mode pair
                mode1 = sim.get("mode_1", "unknown")
                mode2 = sim.get("mode_2", "unknown")
                mode_pair = f"{mode1} vs {mode2}"
                mode_pair_stats[mode_pair]["count"] += 1
                if sim["same_composition"]:
                    mode_pair_stats[mode_pair]["same_comp_count"] += 1
                mode_pair_stats[mode_pair]["volume_diffs"].append(sim["volume_diff_percent"])
                if sim["cell_similarity"] is not None:
                    mode_pair_stats[mode_pair]["cell_similarities"].append(sim["cell_similarity"])
        
        report_lines.extend([
            f"Total Pairwise Comparisons: {total_comparisons}",
            f"Comparisons with Same Composition: {same_comp_count} ({same_comp_count/total_comparisons*100:.1f}%)",
            f"Average Volume Difference: {np.mean(avg_volume_diff):.2f}%",
            f"Average Cell Similarity: {np.mean(avg_cell_similarity):.3f}" if avg_cell_similarity else "",
            "",
        ])
        
        # Mode-specific statistics
        if len(comparison_modes) > 1:
            report_lines.extend([
                "MODE-SPECIFIC STATISTICS:",
                "-" * 80,
            ])
            for mode_pair in sorted(mode_pair_stats.keys()):
                stats = mode_pair_stats[mode_pair]
                report_lines.extend([
                    f"\n{mode_pair}:",
                    f"  Total Comparisons: {stats['count']}",
                    f"  Same Composition: {stats['same_comp_count']} ({stats['same_comp_count']/stats['count']*100:.1f}%)",
                    f"  Avg Volume Difference: {np.mean(stats['volume_diffs']):.2f}%",
                    f"  Avg Cell Similarity: {np.mean(stats['cell_similarities']):.3f}" if stats['cell_similarities'] else "  Avg Cell Similarity: N/A",
                ])
    
    report_lines.extend([
        "",
        "=" * 80,
    ])
    
    # Add mode descriptions
    report_lines.extend([
        "",
        "MODE DESCRIPTIONS:",
        "-" * 80,
        "  text: Composition-based prompt (e.g., 'A crystal structure of K2U1Pd2S2')",
        "  cif: CIF content-based prompt (using actual CIF file text as input)",
        "  text+cif: Combined prompt with both composition and full CIF content",
    ])
    
    report_lines.extend([
        "",
        "=" * 80,
    ])
    
    report_text = "\n".join(report_lines)
    
    report_file = Path(output_dir) / "REPORT.txt"
    with open(report_file, "w") as f:
        f.write(report_text)
    
    print("\n" + report_text)
    print(f"✓ Report saved to {report_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze CIF files with Chemeleon model")
    parser.add_argument("--csv", type=str, required=True,
                        help="Path to CSV file containing CIF file paths")
    parser.add_argument("--output-dir", type=str, default="../results/chemeleon_analysis_results",
                        help="Directory to save results")
    parser.add_argument("--max-cifs", type=int, default=None,
                        help="Maximum number of CIF files to process")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a custom Chemeleon checkpoint (uses default if not specified)")
    parser.add_argument("--clip-checkpoint", type=str, default=None,
                        help="Path to a custom CLIP checkpoint (uses default if not specified)")
    parser.add_argument("--modes", type=str, nargs="+", default=["text", "cif", "text+cif"],
                        choices=["text", "cif", "text+cif"],
                        help="Comparison modes: text (composition), cif (CIF content), text+cif (both)")
    parser.add_argument("--cif-base-dir", type=str, default=None,
                        help="Base directory for relative CIF paths in CSV")
    
    args = parser.parse_args()
    
    print("Starting CIF analysis with Chemeleon...")
    print(f"CSV File: {args.csv}")
    print(f"Output Directory: {args.output_dir}")
    if args.checkpoint:
        print(f"Chemeleon Checkpoint: {args.checkpoint}")
    if args.clip_checkpoint:
        print(f"CLIP Checkpoint: {args.clip_checkpoint}")
    
    process_cif_files(args.csv, args.output_dir, args.max_cifs, args.checkpoint, args.modes, args.clip_checkpoint, args.cif_base_dir)
    
    print("\nAnalysis complete!")
