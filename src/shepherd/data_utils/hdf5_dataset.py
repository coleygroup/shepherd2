"""
Optimized Lazy-loading HDF5 dataset.

This class inherits from HeteroDataset and assumes the parent
provides a `process_item(self, molblock, charges, k)` method.

This version is:
- Memory-efficient: Only loads data from HDF5 on demand.
- Multiprocessing-safe: Handles file opening within each worker process.
- Simple: Indexes all files on init and removes complex/unnecessary file-reloading logic.
"""

import h5py
from pathlib import Path
from shepherd.datasets import HeteroDataset  # Assumes this has been refactored

class HeteroDatasetHDF5(HeteroDataset):
    """
    Dataset that lazily loads molecules from HDF5 files only when __getitem__
    is called. It uses the parent's `process_item` method to perform
    all data processing (noise, graph creation, etc.).
    """

    def __init__(
        self,
        hdf5_data_dir,
        cache_molecules=False,
        **hetero_kwargs  # All the HeteroDataset parameters
    ):
        """
        Args:
            hdf5_files: List of paths to HDF5 files.
            cache_molecules: If True, cache loaded molblocks/charges in RAM.
                             (Trades memory for speed after first epoch).
            **hetero_kwargs: All parameters normally passed to HeteroDataset
        """
        # Initialize the parent class with a dummy empty list.
        # The parent's processing logic (now in `process_item`)
        # will be inherited, but its data list won't be used.
        super().__init__(
            molblocks_and_charges=[],
            **hetero_kwargs
        )

        self.hdf5_data_dir = Path(hdf5_data_dir)
        if not self.hdf5_data_dir.exists() and not self.hdf5_data_dir.is_dir():
            raise FileNotFoundError(f"HDF5 data directory {self.hdf5_data_dir} does not exist")

        self.hdf5_files = sorted(self.hdf5_data_dir.glob('*.h5'))
        if len(self.hdf5_files) == 0:
            raise FileNotFoundError(f"No HDF5 files found in {self.hdf5_data_dir}")

        self.cache_molecules = cache_molecules
        self.molecule_cache = {} if cache_molecules else None

        # This dictionary will store the open HDF5 file handles.
        # CRITICAL: It is initialized empty. Each dataloader worker will populate its *own*
        # version of this dictionary.
        self.open_files = {}

        # Build the index mapping a global `idx` to a specific file and a local index within that file.
        self._build_index()

        # Set the true length of the dataset
        self.length = len(self.index_map)

    def _build_index(self):
        """Build mapping from global index to (file_idx, local_idx)."""
        self.index_map = []  # List of (file_idx, local_idx) tuples
        self.file_sizes = {}

        print(f"Building index for {len(self.hdf5_files)} HDF5 files...")

        for file_idx, file_path in enumerate(self.hdf5_files):
            # Open file just to read metadata (fast)
            try:
                with h5py.File(file_path, 'r') as f:
                    # Assumes metadata 'n_molecules' is stored in file attrs
                    n_molecules = f.attrs['n_molecules']
                    self.file_sizes[file_idx] = n_molecules

                # Add entries to index
                for local_idx in range(n_molecules):
                    self.index_map.append((file_idx, local_idx))
            except Exception as e:
                print(f"Warning: Could not read metadata from {file_path}. Skipping file.")
                print(f"Error: {e}")

        total_size = len(self.index_map)
        print(f"Index built: {total_size:,} total molecules from {len(self.file_sizes)} valid files.")

    def _get_molblock_and_charges(self, idx):
        """
        Load a single molecule from HDF5.
        This method is worker-safe and opens file handles on first access.
        """
        # Check cache first
        if self.cache_molecules and idx in self.molecule_cache:
            return self.molecule_cache[idx]

        # Map global index to file and local index
        file_idx, local_idx = self.index_map[idx]

        # --- Worker-Safe File Opening ---
        # Check if this *worker process* has the file open.
        if file_idx not in self.open_files:
            # If not, this worker opens it and stores the handle.
            # This handle is local to the current worker.
            file_path = self.hdf5_files[file_idx]
            try:
                self.open_files[file_idx] = h5py.File(file_path, 'r', swmr=True)
            except Exception as e:
                # Fallback to non-SWMR mode if it fails
                print(f"Warning: SWMR mode failed for {file_path} (Error: {e}). Opening in regular mode.")
                self.open_files[file_idx] = h5py.File(file_path, 'r')

        # Get the (now guaranteed to be open) HDF5 file handle
        hdf5_file = self.open_files[file_idx]

        # --- End Worker-Safe Logic ---

        # Read ONLY this one molecule's data
        # Assumes HDF5 datasets are named 'molblocks', 'charges_lengths', 'charges'
        molblock = hdf5_file['molblocks'][local_idx]
        if isinstance(molblock, bytes):
            molblock = molblock.decode('utf-8')

        charges_length = int(hdf5_file['charges_lengths'][local_idx])
        # .copy() is important to avoid issues with read-only HDF5 arrays
        charges = hdf5_file['charges'][local_idx, :charges_length].copy()

        result = (molblock, charges)

        # Cache if enabled
        if self.cache_molecules:
            self.molecule_cache[idx] = result

        return result

    def __getitem__(self, k):
        """
        The main entry point for the DataLoader.
        1. Lazily loads raw data from HDF5.
        2. Calls the parent's `process_item` to convert raw data into a graph.
        """
        # 1. Get raw data from HDF5 using our lazy, worker-safe method
        molblock, charges = self._get_molblock_and_charges(k)

        # 2. Call the inherited processing logic from the parent class
        return self.process_item(molblock, charges, k)

    def __len__(self):
        """Return the total number of molecules in the index."""
        return self.length

    # --- PyTorch Geometric Compatibility ---
    # (If your parent class doesn't already have these)

    def len(self):
        return self.__len__()

    def get(self, idx):
        return self.__getitem__(idx)
