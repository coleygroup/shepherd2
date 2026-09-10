import pickle
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

def convert_pickle_to_hdf5(pickle_path, hdf5_path):
    """
    Convert pickle file with List[Tuple[str, np.array]] to HDF5.

    HDF5 structure:
    - /molblocks: variable-length string dataset
    - /charges: dataset with shape (n_molecules, max_atoms)
    - /charges_lengths: actual number of atoms per molecule (for masking)
    """
    print(f"Converting {pickle_path} to {hdf5_path}")

    # Load pickle data
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)  # List[Tuple[str, np.array]]

    n_molecules = len(data)

    # Extract molblocks and charges
    molblocks = [item[0] for item in data]
    charges_list = [item[1] for item in data]

    # Find max number of atoms (for padding charges array)
    max_atoms = max(len(charges) for charges in charges_list)
    charges_lengths = np.array([len(charges) for charges in charges_list], dtype=np.int32)

    # Pad charges to same length
    charges_padded = np.zeros((n_molecules, max_atoms), dtype=np.float32)
    for i, charges in enumerate(charges_list):
        charges_padded[i, :len(charges)] = charges

    # Write to HDF5
    with h5py.File(hdf5_path, 'w') as f:
        # Store molblocks as variable-length strings
        dt = h5py.string_dtype(encoding='utf-8')
        molblocks_dset = f.create_dataset(
            'molblocks',
            (n_molecules,),
            dtype=dt,
            compression='gzip',  # Compress text data
            compression_opts=4
        )
        molblocks_dset[:] = molblocks

        # Store charges as float array
        f.create_dataset(
            'charges',
            data=charges_padded,
            compression='gzip',
            compression_opts=4,
            chunks=(min(1000, n_molecules), max_atoms)  # Chunk for better random access
        )

        # Store actual lengths
        f.create_dataset('charges_lengths', data=charges_lengths)

        # Store metadata
        f.attrs['n_molecules'] = n_molecules
        f.attrs['max_atoms'] = max_atoms

    print(f"Saved {n_molecules} molecules to {hdf5_path}")

    # Report size reduction
    import os
    pickle_size = os.path.getsize(pickle_path) / (1024**3)  # GB
    hdf5_size = os.path.getsize(hdf5_path) / (1024**3)  # GB
    print(f"Size: {pickle_size:.2f} GB (pickle) -> {hdf5_size:.2f} GB (HDF5)")
    print(f"Compression ratio: {pickle_size/hdf5_size:.2f}x")


def load_hdf5_to_list(hdf5_path):
    """
    Load HDF5 file back to List[Tuple[str, np.array]] format in the same order.

    Args:
        hdf5_path: Path to HDF5 file

    Returns:
        List[Tuple[str, np.array]]: List of (molblock, charges) tuples
    """
    with h5py.File(hdf5_path, 'r') as f:
        # Read all molblocks
        molblocks = f['molblocks'][:]
        # Decode bytes to strings if needed
        if isinstance(molblocks[0], bytes):
            molblocks = [mb.decode('utf-8') for mb in molblocks]
        else:
            molblocks = list(molblocks)

        # Read all charges (padded) and lengths
        charges_padded = f['charges'][:]  # Shape: (n_molecules, max_atoms)
        charges_lengths = f['charges_lengths'][:]  # Shape: (n_molecules,)

        # Reconstruct original charges arrays by slicing
        data = []
        for i in range(len(molblocks)):
            charges_length = int(charges_lengths[i])
            charges = charges_padded[i, :charges_length].copy()
            data.append((molblocks[i], charges))

    return data


def convert_all_files(data_dir, output_dir):
    """Convert all pickle files in a directory to HDF5."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pickle_files = sorted(data_dir.glob('molblock_charges_*.pkl'))
    print(f"Found {len(pickle_files)} pickle files")

    for pickle_path in tqdm(pickle_files, desc="Converting files"):
        hdf5_path = output_dir / f"{pickle_path.stem}.h5"
        if not hdf5_path.exists():
            convert_pickle_to_hdf5(pickle_path, hdf5_path)
        else:
            print(f"Skipping {hdf5_path} (already exists)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    convert_all_files(
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    print("Done")