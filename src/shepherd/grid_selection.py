"""Grid-affinity-based atom subselection

Scores ligand atoms against precomputed AutoDock affinity grids for two targets and
selects atoms whose interaction is differentially favorable. The returned atom indices can be input into
:meth:`shepherd.interaction_profile.InteractionProfile.subselection`.
"""
from __future__ import annotations

import os

import numpy as np
import torch
from rdkit import Chem
from meeko import MoleculePreparation

XS_DICT = {
    'C_H': 0,
    'C_P': 1,
    'N_P': 2,
    'N_D': 3,
    'N_A': 4,
    'N_DA': 5,
    'O_P': 6,
    'O_D': 7,
    'O_A': 8,
    'O_DA': 9,
    'S_P': 10,
    'P_P': 11,
    'F_H': 12,
    'Cl_H': 13,
    'Br_H': 14,
    'I_H': 15,
    'Met_D': 16,
}


def _bonded_to_heteroatom(atom: Chem.Atom) -> bool:
    return any(n.GetAtomicNum() not in (6, 1) for n in atom.GetNeighbors())


def _bonded_to_H(atom: Chem.Atom) -> bool:
    return any(n.GetAtomicNum() == 1 for n in atom.GetNeighbors())


def assign_xs_types(mol: Chem.Mol) -> dict[int, int | None]:
    """Assign an AutoDock XS type (see ``XS_DICT``) to every atom in ``mol``."""
    prep = MoleculePreparation()
    setup = prep.prepare(mol)
    ad_types = [atom.atom_type for atom in setup[0].atoms]

    xs_types: dict[int, int | None] = {}
    for atom in mol.GetAtoms():
        element = atom.GetAtomicNum()
        idx = atom.GetIdx()
        ad = ad_types[idx] if idx < len(ad_types) else None

        acceptor = ad in ("OA", "NA")
        donor_n_or_o = element == 10 or _bonded_to_H(atom)

        if element == 6:  # Carbon
            xs_types[idx] = XS_DICT['C_P'] if _bonded_to_heteroatom(atom) else XS_DICT['C_H']
        elif element == 7:  # Nitrogen
            if acceptor and donor_n_or_o:
                xs_types[idx] = XS_DICT['N_DA']
            elif acceptor:
                xs_types[idx] = XS_DICT['N_A']
            elif donor_n_or_o:
                xs_types[idx] = XS_DICT['N_D']
            else:
                xs_types[idx] = XS_DICT['N_P']
        elif element == 8:  # Oxygen
            if acceptor and donor_n_or_o:
                xs_types[idx] = XS_DICT['O_DA']
            elif acceptor:
                xs_types[idx] = XS_DICT['O_A']
            elif donor_n_or_o:
                xs_types[idx] = XS_DICT['O_D']
            else:
                xs_types[idx] = XS_DICT['O_P']
        elif element == 16:  # Sulfur
            xs_types[idx] = XS_DICT['S_P']
        elif element == 15:  # Phosphorus
            xs_types[idx] = XS_DICT['P_P']
        elif element == 9:  # Fluorine
            xs_types[idx] = XS_DICT['F_H']
        elif element == 17:  # Chlorine
            xs_types[idx] = XS_DICT['Cl_H']
        elif element == 35:  # Bromine
            xs_types[idx] = XS_DICT['Br_H']
        elif element == 53:  # Iodine
            xs_types[idx] = XS_DICT['I_H']
        else:
            xs_types[idx] = None

    return xs_types


def read_autodock_map(filename: str):
    """Parse an AutoDock ``.map`` affinity grid file.

    Returns ``(spacing, nx, ny, nz, center, data)`` where ``data`` is an
    ``(nx+1, ny+1, nz+1)`` array of grid energies.
    """
    with open(filename, "rb") as f:
        header_lines = []
        for _ in range(6):
            pos = f.tell()
            line = f.readline()
            try:
                header_lines.append(line.decode("utf-8").strip())
            except UnicodeDecodeError:
                f.seek(pos)
                break

    spacing = None
    nx = ny = nz = None
    center = [0.0, 0.0, 0.0]
    for line in header_lines:
        toks = line.split()
        if not toks:
            continue
        if toks[0] == "SPACING":
            spacing = float(toks[1])
        elif toks[0] == "NELEMENTS":
            nx, ny, nz = map(int, toks[1:4])
        elif toks[0] == "CENTER":
            center = list(map(float, toks[1:4]))

    if None in (spacing, nx, ny, nz):
        raise RuntimeError(f"Failed to parse AutoDock map header: {filename}")

    with open(filename, "rb") as f:
        for _ in header_lines:
            f.readline()
        data = np.array([float(x) for x in f.read().splitlines() if x.strip() != ""])
        data = data.reshape((nx + 1, ny + 1, nz + 1), order='F')

    return spacing, nx, ny, nz, center, data


def convert_pos_coor(
    pos: torch.Tensor,
    spacing: float,
    nx: int,
    ny: int,
    nz: int,
    center: list,
) -> torch.Tensor:
    """Convert world-space positions to continuous grid-voxel coordinates."""
    return (pos - torch.tensor(center, device=pos.device)) / spacing + torch.tensor(
        [nx / 2, ny / 2, nz / 2], device=pos.device
    )


def trilinear_interp(grid: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    """Interpolate scalar ``grid`` values at continuous voxel coordinates ``pts``.

    ``grid``: (N, N, N) tensor of scalar values.
    ``pts``: (P, 3) tensor of continuous grid coordinates, 0 <= x,y,z <= N-1.
    Returns an (P,) tensor of interpolated values.
    """
    N = grid.shape[0]
    x, y, z = pts.unbind(-1)

    i0 = torch.clamp(x.floor().long(), 0, N - 2)
    j0 = torch.clamp(y.floor().long(), 0, N - 2)
    k0 = torch.clamp(z.floor().long(), 0, N - 2)

    tx = (x - i0).clamp(0, 1)
    ty = (y - j0).clamp(0, 1)
    tz = (z - k0).clamp(0, 1)

    g000 = grid[i0, j0, k0]
    g100 = grid[i0 + 1, j0, k0]
    g010 = grid[i0, j0 + 1, k0]
    g110 = grid[i0 + 1, j0 + 1, k0]
    g001 = grid[i0, j0, k0 + 1]
    g101 = grid[i0 + 1, j0, k0 + 1]
    g011 = grid[i0, j0 + 1, k0 + 1]
    g111 = grid[i0 + 1, j0 + 1, k0 + 1]

    return (
        (1 - tx) * (1 - ty) * (1 - tz) * g000 +
        tx * (1 - ty) * (1 - tz) * g100 +
        (1 - tx) * ty * (1 - tz) * g010 +
        tx * ty * (1 - tz) * g110 +
        (1 - tx) * (1 - ty) * tz * g001 +
        tx * (1 - ty) * tz * g101 +
        (1 - tx) * ty * tz * g011 +
        tx * ty * tz * g111
    )


class Grid:
    """A single AutoDock affinity grid for one XS atom type."""

    def __init__(self, grid_file: str):
        self.spacing, self.nx, self.ny, self.nz, self.center, grid_data = read_autodock_map(grid_file)
        self.grid_tensor = torch.tensor(grid_data, dtype=torch.float32)

    def get_coor(self, positions: torch.Tensor) -> torch.Tensor:
        return convert_pos_coor(positions, self.spacing, self.nx, self.ny, self.nz, self.center)

    def get_energies(self, positions: torch.Tensor) -> torch.Tensor:
        return trilinear_interp(grid=self.grid_tensor, pts=self.get_coor(positions))


class Receptor:
    """A directory of per-XS-type AutoDock affinity grids for one target."""

    def __init__(self, grid_dir: str):
        self.grids: list[Grid | None] = [None] * len(XS_DICT)

        files = [os.path.join(grid_dir, f) for f in os.listdir(grid_dir)]
        for file in files:
            if not file.endswith('.map'):
                raise ValueError(f"All files in grid_dir must be .map files. Found: {file}")
            for atom_type, xs_idx in XS_DICT.items():
                if atom_type in file:
                    self.grids[xs_idx] = Grid(file)

    def get_energies(self, mol: Chem.Mol) -> list[float]:
        """Get the affinity-grid energy for every heavy atom in ``mol``."""
        xs_types = assign_xs_types(mol)
        energies = []

        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() <= 1:
                continue
            idx = atom.GetIdx()
            pos = mol.GetConformer().GetAtomPosition(idx)
            position_tensor = torch.tensor([[pos.x, pos.y, pos.z]], dtype=torch.float32)

            xs_type = xs_types[idx]
            grid = self.grids[xs_type]
            if grid is None:
                raise ValueError(f"No grid found for XS type {xs_type}")

            energies.append(grid.get_energies(position_tensor).item())

        return energies


def reorder_mol_to_match(mol: Chem.Mol, ref: Chem.Mol) -> Chem.Mol:
    """Renumber ``mol``'s atoms to match ``ref``'s atom order via substructure match."""
    match = mol.GetSubstructMatch(ref)
    if not match:
        raise ValueError("Molecules do not match")
    return Chem.RenumberAtoms(mol, list(match))


class GridEnergyDiffSelector:
    """Grid-affinity-based atom selection."""

    def __init__(self, mol_on: Chem.Mol, mol_off: Chem.Mol, grid_dir_on: str, grid_dir_off: str):
        self.rec_on = Receptor(grid_dir_on)
        self.rec_off = Receptor(grid_dir_off)
        self.mol_on = reorder_mol_to_match(mol_on, mol_off)
        self.mol_off = mol_off

    def get_diff_energies(self) -> np.ndarray:
        """Per-heavy-atom energy difference: energy at ``grid_dir_off`` minus
        energy at ``grid_dir_on``, in ``mol_off`` heavy-atom order."""
        energies_on = np.array(self.rec_on.get_energies(self.mol_on))
        energies_off = np.array(self.rec_off.get_energies(self.mol_off))
        return energies_off - energies_on

    def choose_atoms(self, threshold: float) -> np.ndarray:
        """Heavy-atom indices into ``mol_off`` where the energy difference is below
        ``threshold``."""
        diff_energies = self.get_diff_energies()
        heavy_atom_inds = np.array(
            [atom.GetIdx() for atom in self.mol_off.GetAtoms() if atom.GetAtomicNum() > 1]
        )
        return heavy_atom_inds[diff_energies < threshold]
