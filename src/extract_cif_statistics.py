#!/usr/bin/env python3
"""
Extract detailed statistics from all CIFs in a tarball
"""

import tarfile
import sys
import pandas as pd
from pathlib import Path
import tempfile
import re
from pymatgen.core import Composition, Structure
from pymatgen.io.cif import CifParser

def extract_cif_statistics(cif_str, filename):
    """Extract statistics from a single CIF string."""
    stats = {
        'filename': filename,
        'size_bytes': len(cif_str),
        'valid_format': False
    }
    
    try:
        # Parse CIF
        parser = CifParser.from_str(cif_str)
        cif_data = parser.as_dict()
        
        if not cif_data:
            return stats
        
        stats['valid_format'] = True
        
        # Get first structure key
        key = list(cif_data.keys())[0]
        data = cif_data[key]
        
        # Extract formula info
        if '_chemical_formula_sum' in data:
            stats['formula_sum'] = data['_chemical_formula_sum']
            try:
                comp = Composition(stats['formula_sum'])
                stats['num_elements'] = len(comp.elements)
                stats['num_atoms'] = int(comp.num_atoms)
            except:
                pass
        
        if '_chemical_formula_structural' in data:
            stats['formula_structural'] = data['_chemical_formula_structural']
        
        # Extract crystal system
        if '_symmetry_space_group_name_H-M' in data:
            stats['space_group'] = data['_symmetry_space_group_name_H-M']
        
        if '_symmetry_Int_Tables_number' in data:
            try:
                stats['space_group_number'] = int(data['_symmetry_Int_Tables_number'])
            except:
                pass
        
        # Extract cell parameters
        if '_cell_length_a' in data:
            try:
                stats['cell_a'] = float(data['_cell_length_a'])
                stats['cell_b'] = float(data['_cell_length_b'])
                stats['cell_c'] = float(data['_cell_length_c'])
                stats['cell_alpha'] = float(data['_cell_angle_alpha'])
                stats['cell_beta'] = float(data['_cell_angle_beta'])
                stats['cell_gamma'] = float(data['_cell_angle_gamma'])
                stats['cell_volume'] = float(data['_cell_volume'])
            except:
                pass
        
        # Extract atom count
        if '_atom_site_type_symbol' in data and '_atom_site_symmetry_multiplicity' in data:
            try:
                atom_types = data['_atom_site_type_symbol']
                multiplicities = [int(m) for m in data['_atom_site_symmetry_multiplicity']]
                stats['num_atom_sites'] = len(atom_types)
                stats['total_atoms_by_multiplicity'] = sum(multiplicities)
            except:
                pass
        
        # Extract Z (formula units per cell)
        if '_cell_formula_units_Z' in data:
            try:
                stats['Z'] = int(data['_cell_formula_units_Z'])
            except:
                pass
        
        # Try to parse as Structure
        try:
            structure = parser.get_structures()[0]
            stats['structure_valid'] = True
            stats['nsites'] = len(structure)
            stats['composition'] = str(structure.composition)
            stats['density'] = structure.density
        except:
            stats['structure_valid'] = False
    
    except Exception as e:
        stats['error'] = str(e)[:100]
    
    return stats

def generate_cif_statistics(tar_path, output_csv=None):
    """Generate statistics for all CIFs in a tarball."""
    
    print(f"Reading CIFs from {tar_path}...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Extract all CIFs
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(tmpdir)
        
        cif_files = sorted(list(Path(tmpdir).glob('*.cif')))
        print(f"Found {len(cif_files)} CIF files\n")
        
        all_stats = []
        
        for i, cif_file in enumerate(cif_files, 1):
            print(f"[{i:3d}/{len(cif_files)}] Processing {cif_file.name}...", end=" ", flush=True)
            
            try:
                with open(cif_file, 'r') as f:
                    cif_str = f.read()
                
                stats = extract_cif_statistics(cif_str, cif_file.name)
                all_stats.append(stats)
                
                if stats.get('valid_format'):
                    print(f"✓ ({stats.get('num_atoms', '?')} atoms)")
                else:
                    print(f"✗ (parse error)")
            except Exception as e:
                print(f"! Error: {str(e)[:50]}")
                all_stats.append({
                    'filename': cif_file.name,
                    'error': str(e)[:100]
                })
        
        # Convert to DataFrame
        df = pd.DataFrame(all_stats)
        
        print(f"\n{'='*80}")
        print(f"STATISTICS SUMMARY")
        print(f"{'='*80}\n")
        
        # Basic statistics
        print(f"Total CIFs: {len(df)}")
        print(f"Valid format: {df['valid_format'].sum()}")
        print(f"Structure parseable: {df['structure_valid'].sum()}")
        
        # Numeric statistics
        numeric_cols = ['num_atoms', 'num_elements', 'cell_volume', 'density', 'Z']
        for col in numeric_cols:
            if col in df.columns:
                valid_data = df[col].dropna()
                if len(valid_data) > 0:
                    print(f"\n{col}:")
                    print(f"  Mean: {valid_data.mean():.2f}")
                    print(f"  Median: {valid_data.median():.2f}")
                    print(f"  Min: {valid_data.min():.2f}")
                    print(f"  Max: {valid_data.max():.2f}")
        
        # Composition analysis
        if 'formula_sum' in df.columns:
            print(f"\nFormula examples:")
            for idx, row in df[['filename', 'formula_sum']].head(10).iterrows():
                print(f"  {row['filename']}: {row['formula_sum']}")
        
        # Save to CSV if requested
        if output_csv:
            df.to_csv(output_csv, index=False)
            print(f"\n✓ Saved statistics to {output_csv}")
        
        # Display DataFrame
        print(f"\n{'='*80}")
        print("Full Statistics Table (first 20 rows):")
        print(f"{'='*80}\n")
        print(df.head(20).to_string())
        
        return df

if __name__ == '__main__':
    tar_file = 'CrystaLLM/gen_v1_small_raw.tar.gz'
    output_csv = 'cif_statistics.csv'
    
    df = generate_cif_statistics(tar_file, output_csv)
    
    print(f"\n{'='*80}")
    print(f"Statistics saved to: {output_csv}")
    print(f"{'='*80}\n")
