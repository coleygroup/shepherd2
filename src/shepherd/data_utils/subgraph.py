import math
import random
import numpy as np
from rdkit import Chem
from rdkit.Chem import BRICS

MAX_SUBGRAPH_BONDS = 6

def bond_inds_to_atom_inds(mol: Chem.Mol, bond_inds: list[int]) -> list[int]:
    atom_inds = set()
    for bid in bond_inds:
        bond = mol.GetBondWithIdx(bid)
        atom_inds.add(bond.GetBeginAtomIdx())
        atom_inds.add(bond.GetEndAtomIdx())
    return sorted(atom_inds)


def get_subgraph_inds(mol: Chem.Mol, upper_bound_fraction: float = 0.2) -> list[int]:
    """
    Get a random subgraph of the molecule with a number of bonds between 1 and
    ``upper_bound_fraction * num_bonds``.

    The enumerated chain length is capped at ``MAX_SUBGRAPH_BONDS`` so large molecules
    do not trigger full enumeration of very long paths.

    Returns a list of **atom** indices corresponding to the subgraph.
    """
    max_size = max(2, math.ceil(mol.GetNumBonds() * upper_bound_fraction))
    max_size = max(2, min(max_size, MAX_SUBGRAPH_BONDS))
    size_n = np.random.choice(np.arange(1, max_size), 1)[0]
    bond_inds_list = Chem.FindAllSubgraphsOfLengthN(mol, int(size_n), useHs=True)
    bond_ind_idx = int(np.random.choice(len(bond_inds_list), 1)[0])
    atom_inds_list = bond_inds_to_atom_inds(mol, bond_inds_list[bond_ind_idx])
    return atom_inds_list


def get_multiple_subgraph_inds(mol: Chem.Mol, upper_bound_fractions: list[float]) -> list[int]:
    atom_inds_list = []
    for frac in upper_bound_fractions:
        atom_inds_list.extend(get_subgraph_inds(mol, frac))
    return sorted(list(set(atom_inds_list)))


def sample_fracs(num_fracs: int = 4, options: list[float] = [0.1, 0.2, 0.3]) -> list[float]:
    options_ = [None] + options
    fracs = [i for i in np.random.choice(options_, num_fracs, replace=True) if i is not None]
    if not fracs:
        fracs = [max(options)]
    return fracs


def sample_random_or_subgraphs(
    mol: Chem.Mol,
    prob_random_vs_subgraphs: float = 0.5,
    frac_random: float = 0.1,
    hydrogen_weight: float = 0.3,
    num_fracs: int = 4,
    upper_bound_fraction_options: list[float] = [0.1, 0.2, 0.3],
) -> list[int]:
    """
    Sample either random atoms or subgraphs of the molecule.

    Arguments
    ---------
    mol: Chem.Mol
    prob_random_vs_subgraphs: float (default = 0.5)
        Probability of sampling random atoms instead of subgraphs.
    frac_random: float (default = 0.1)
        Max number of fraction of atoms to sample if random atoms are sampled.
    hydrogen_weight: float (default = 0.3)
        Weight of hydrogens in the probability distribution for random atoms.
    num_fracs: int (default = 4)
        Max number of subgraphs to sample if subgraphs are sampled.
    upper_bound_fraction_options: list[float] (default = [0.1, 0.2, 0.3])
        Options for the upper bound fraction of the subgraphs.

    Returns
    -------
    atom_inds: list[int]
        Atom indices of the subgraph or random atoms.
    """
    if np.random.uniform(0, 1) < prob_random_vs_subgraphs:
        fracs = sample_fracs(num_fracs, upper_bound_fraction_options)
        atom_inds = get_multiple_subgraph_inds(mol, fracs)
    else:
        # downweight hydrogens getting selected
        h_inds = np.array([a.GetAtomicNum() == 1 for a in mol.GetAtoms()])
        n_h = h_inds.sum()
        n_nonh = mol.GetNumAtoms() - n_h
        if n_h == 0:
            p = np.ones(mol.GetNumAtoms()) / mol.GetNumAtoms()
        elif n_nonh == 0:
            p = np.ones(mol.GetNumAtoms()) / mol.GetNumAtoms()
        else:
            p = np.where(
                h_inds == 1,
                hydrogen_weight / n_h,
                (1 - hydrogen_weight) / n_nonh
            )
            p = p / p.sum()

        num_atoms_to_sample = max(1, math.ceil(mol.GetNumAtoms()*frac_random))
        atom_inds = np.random.choice(
            mol.GetNumAtoms(),
            num_atoms_to_sample,
            replace=False,
            p=p
        ).tolist()
    return sorted(atom_inds)


def select_brics_scaffold_indices(mol: Chem.Mol, seed: int) -> list[int]:
    """Selects BRICS scaffold atoms at random (min 10 atoms)"""
    n_original = mol.GetNumAtoms()
    fragments = Chem.GetMolFrags(BRICS.BreakBRICSBonds(mol), asMols=False)
    fragments = [
        tuple(index for index in fragment if index < n_original)
        for fragment in fragments
    ]
    fragments = [fragment for fragment in fragments if fragment]
    if not fragments:
        return []
    largest = max(map(len, fragments))
    candidates = [
        fragment for fragment in fragments if len(fragment) >= min(10, largest)
    ]
    return list(random.Random(seed).choice(candidates))


def _neighbor_hydrogens(mol: Chem.Mol, atom_index: int) -> list[int]:
    return [
        neighbor.GetIdx()
        for neighbor in mol.GetAtomWithIdx(atom_index).GetNeighbors()
        if neighbor.GetAtomicNum() == 1
    ]


def _expanded_atom_group(mol: Chem.Mol, atom_index: int) -> tuple[list[int], list[int]]:
    """Terminal-neighbor expansion to a single heavy atom"""
    heavy = [atom_index]
    atom = mol.GetAtomWithIdx(atom_index)
    neighbors = list(atom.GetNeighbors())
    if len(neighbors) == 1:
        neighbor_index = neighbors[0].GetIdx()
        if neighbor_index not in heavy:
            heavy.append(neighbor_index)
    elif atom.GetAtomicNum() == 6:
        for heteroatom in neighbors:
            if heteroatom.GetAtomicNum() <= 1 or heteroatom.GetAtomicNum() == 6:
                continue
            heavy_neighbors = [
                neighbor
                for neighbor in heteroatom.GetNeighbors()
                if neighbor.GetAtomicNum() > 1
            ]
            if len(heavy_neighbors) == 1 and heteroatom.GetIdx() not in heavy:
                heavy.append(heteroatom.GetIdx())
    hydrogens = []
    for index in heavy:
        hydrogens.extend(_neighbor_hydrogens(mol, index))
    return heavy, hydrogens


def select_hetero_scaffold_indices(mol: Chem.Mol, seed: int) -> list[int]:
    """Select hetero/non-ring-carbon atoms and attached atoms"""
    candidates = [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() > 1
        and (atom.GetAtomicNum() != 6 or not atom.IsInRing())
    ]
    if not candidates:
        return []
    rng = random.Random(seed)
    selected = rng.sample(candidates, min(rng.randint(1, 3), len(candidates)))
    indices: list[int] = []
    for atom_index in selected:
        heavy, hydrogens = _expanded_atom_group(mol, atom_index)
        indices.extend(heavy)
        indices.extend(hydrogens)
    return sorted(set(indices))
