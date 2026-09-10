"""
Interaction profile extraction and representation.

An InteractionProfile holds the surface point cloud, electrostatic potential (ESP),
and pharmacophore features.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from typing import Literal, TYPE_CHECKING
import logging
import numpy as np
from rdkit import Chem

if TYPE_CHECKING:
    from shepherd.generated_sample import GeneratedSample

from shepherd_score.score.constants import COULOMB_SCALING
from shepherd_score.container import Molecule
from shepherd_score.conformer_generation import (
    update_mol_coordinates,
    charges_from_single_point_conformer_with_xtb,
    optimize_conformer_with_xtb,
)


_MODALITY_FLAGS = {
    'surface': 'inpaint_x3_pos',
    'electrostatics': 'inpaint_x3_x',
    'pharm_positions': 'inpaint_x4_pos',
    'pharm_directions': 'inpaint_x4_direction',
    'pharm_type': 'inpaint_x4_type',
}

_MODALITY_ALIASES = {
    'shape': 'surface',
    'surf': 'surface',
    'esp': 'electrostatics',
    'pharm': 'pharmacophores',
    'pharm_pos': 'pharm_positions',
    'pharm_direction': 'pharm_directions',
    'pharm_ancs': 'pharm_positions',
    'pharm_vecs': 'pharm_directions',
    'x2': 'surface',
    'x3': 'electrostatics',
    'x4': 'pharmacophores',
}

_MODALITY_GROUPS = {
    'pharmacophores': {'pharm_positions', 'pharm_directions', 'pharm_type'},
    'electrostatics': {'surface', 'electrostatics'},
    'x2_x4': {'surface', 'pharm_positions', 'pharm_directions', 'pharm_type'},
}


def _resolve_modalities(condition_modalities: str | set[str]) -> set[str]:
    """Resolve a `condition_modalities` argument to a set of canonical names."""
    if isinstance(condition_modalities, str):
        condition_modalities = condition_modalities.lower()
        if condition_modalities == 'all':
            return set(_MODALITY_FLAGS)
        if (
            condition_modalities in _MODALITY_FLAGS
            or condition_modalities in _MODALITY_ALIASES
            or condition_modalities in _MODALITY_GROUPS
        ):
            condition_modalities = {condition_modalities}
        else:
            raise ValueError(
                f"condition_modalities must be 'all' or a set of modality names, got "
                f"the string {condition_modalities!r}. Did you mean {{{condition_modalities!r}}}?"
            )

    resolved = set()
    for name in condition_modalities:
        name = _MODALITY_ALIASES.get(name.lower(), name.lower())
        resolved |= _MODALITY_GROUPS.get(name, {name})

    invalid = resolved - set(_MODALITY_FLAGS)
    if invalid:
        raise ValueError(
            f"Unknown condition_modalities name(s): {sorted(invalid)}. "
            f"Valid names are {sorted(_MODALITY_FLAGS)}, aliases {sorted(_MODALITY_ALIASES)}, "
            f"and groups {sorted(_MODALITY_GROUPS)}."
        )
    return resolved


@dataclass
class ConditionAtoms:
    """Subset of atoms to condition on for generation.

    Supply one of the two:
    - ``inds`` and ``mol`` (can use ``.from_mol()``)
    - ``types``, ``pos``, and ``formal_charges``.

    Attributes
    ----------
    inds : list[int]
        Indices of the atoms to condition on.
    types : list[int] | None
        Atom types corresponding to ``inds``.
    pos : np.ndarray | None
        Atom positions corresponding to ``inds``.
    formal_charges : np.ndarray | None
        Atom formal charges corresponding to ``inds``.
    exit_vector_inds : list[int] | None
        Optional for atom-inpainting only (not fixed conditioning). The indices of
        the atoms within ``inds``to use as "exit vectors" for inpainting any "bond"
        between inpainted atoms, but excludes bonds between non-inpainted atoms and
        exit vector atoms.
    """
    inds: list[int]
    types: list[int] | None = None
    pos: np.ndarray | None = None
    formal_charges: np.ndarray | None = None
    exit_vector_inds: list[int] | None = None

    @classmethod
    def from_mol(
        cls,
        mol: Chem.Mol,
        inds: list[int],
        exit_vector_inds: list[int] | None = None,
    ) -> 'ConditionAtoms':
        """Fill types, positions, and charges from ``mol`` at ``inds``."""
        atom_inds = list(inds)
        coords = np.asarray(mol.GetConformer().GetPositions())
        return cls(
            inds=atom_inds,
            types=[int(mol.GetAtomWithIdx(i).GetAtomicNum()) for i in atom_inds],
            pos=coords[np.array(atom_inds)],
            formal_charges=np.array(
                [mol.GetAtomWithIdx(i).GetFormalCharge() for i in atom_inds]
            ),
            exit_vector_inds=(
                None if exit_vector_inds is None else list(exit_vector_inds)
            ),
        )

    def to_dict(self) -> dict:
        return {
            'inds': list(self.inds),
            'types': None if self.types is None else list(self.types),
            'pos': self.pos,
            'formal_charges': self.formal_charges,
            'exit_vector_inds': (
                None if self.exit_vector_inds is None else list(self.exit_vector_inds)
            ),
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'ConditionAtoms':
        return cls(
            inds=list(d['inds']),
            types=None if d.get('types') is None else list(d['types']),
            pos=None if d.get('pos') is None else np.asarray(d['pos']),
            formal_charges=(
                None if d.get('formal_charges') is None
                else np.asarray(d['formal_charges'])
            ),
            exit_vector_inds=(
                None if d.get('exit_vector_inds') is None
                else list(d['exit_vector_inds'])
            ),
        )

    def __len__(self) -> int:
        return len(self.inds)


@dataclass
class InteractionProfile:
    """Interaction profile: consists of surface point cloud,
    electrostatic potential computed at each surface point, and pharmacophore features.

    `com_before_centering` is the center of mass (COM) before centering which
    can be used to recover the original positions. Assumes the profiles are already centered.

    Attributes
    ----------
    surface : np.ndarray | None (M, 3)
        Surface point cloud coordinates.
    electrostatics : np.ndarray | None (M,)
        Electrostatic potential at each surface point.
    partial_charges : np.ndarray | None (n_atoms,)
        Atomic partial charges (incl. H) used to compute ``electrostatics``,
        if known (e.g. from xTB).
    pharm_types : np.ndarray | None (P,)
        Integer pharmacophore type labels.
    pharm_positions : np.ndarray | None (P, 3)
        Pharmacophore anchor positions.
    pharm_direction : np.ndarray | None (P, 3)
        Pharmacophore direction unit vectors.
    com_before_centering : np.ndarray (3,)
        Mean atom position before coords were shifted to the origin.
    smiles : str | None
        SMILES string of the source molecule, if known.
    mol : Chem.Mol | None
        An rdkit molecule used to extract the interaction profile, if known.
    condition_atoms : ConditionAtoms | None
        Subset of atoms to condition on (x1 inpainting or scaffold).
        Indices refer to ``mol``.
    neutralize_esp : bool
        If True, ``electrostatics`` was (or should be) computed with net
        charge smeared uniformly over atoms so ``sum(q) == 0``. Applied at
        extraction time via :func:`compute_neutralized_esp`.
    """
    surface: np.ndarray | None = None
    electrostatics: np.ndarray | None = None
    partial_charges: np.ndarray | None = None
    pharm_types: np.ndarray | None = None
    pharm_positions: np.ndarray | None = None
    pharm_directions: np.ndarray | None = None
    com_before_centering: np.ndarray = field(default_factory=lambda: np.zeros(3))
    smiles: str | None = None
    mol: Chem.Mol | None = None
    n_atoms: int = 0 # total number of atoms (incl. H) in the source molecule
    n_pharms: int = field(init=False)  # always derived from pharm_types
    condition_atoms: ConditionAtoms | None = None
    neutralize_esp: bool = False

    def __post_init__(self):
        self.n_pharms = 0 if self.pharm_types is None else len(self.pharm_types)

    def _present_modalities(self) -> set:
        present = set()
        if self.surface is not None:
            present.add('surface')
        if self.electrostatics is not None:
            present.add('electrostatics')
        if self.pharm_positions is not None:
            present.add('pharm_positions')
        if self.pharm_directions is not None:
            present.add('pharm_directions')
        if self.pharm_types is not None:
            present.add('pharm_type')
        return present

    def to_molecule(
        self,
        num_surf_points: int = 400,
        probe_radius: float = 1.2,
        pharm_multi_vector: bool | None = False,
        partial_charges: np.ndarray | None = None,
    ) -> Molecule:
        """Build a shepherd-score Molecule object for evaluation.

        Evaluation with shepherd-score uses 400 surface points and a probe radius of 1.2,
        which differ from the generation defaults of 75 and 0.6, respectively.
        """
        if self.mol is None:
            raise ValueError(
                'InteractionProfile.mol is required to build a shepherd_score Molecule'
            )
        reuse_surface = (
            self.surface is not None and len(self.surface) == num_surf_points
        )
        reuse_pharm = (
            self.pharm_types is not None
            and self.pharm_positions is not None
            and self.pharm_directions is not None
        )
        if partial_charges is None:
            partial_charges = self.partial_charges
        return Molecule(
            self.mol,
            num_surf_points=None if reuse_surface else num_surf_points,
            probe_radius=probe_radius,
            surface_points=self.surface if reuse_surface else None,
            electrostatics=self.electrostatics if reuse_surface else None,
            partial_charges=partial_charges,
            pharm_multi_vector=pharm_multi_vector,
            pharm_types=self.pharm_types if reuse_pharm else None,
            pharm_ancs=self.pharm_positions if reuse_pharm else None,
            pharm_vecs=self.pharm_directions if reuse_pharm else None,
        )

    def to_generate_kwargs(self, condition_modalities: str | set[str] = 'all') -> dict:
        """Construct kwargs for `generate()`.

        `condition_modalities` selects which modalities to condition on:
        - 'all' (default): inpaint every channel that is present on this profile
          (missing surface / ESP / pharmacophore fields are skipped).

        Alternatively, specify with a single string or a list/set of strings:
        - 'surface' | 'shape' | 'surf' | 'x2'
        - 'electrostatics' | 'esp' | 'x3'
            Note that esp implies surface since ESP is computed at surface point locations.
        - 'pharmacophores' | 'pharm' | 'x4'
            Inpaints all positions, directions, and types.
            Alternatively, specify individual sub-fields: 'pharm_positions',
            'pharm_directions', 'pharm_type' (aliases: 'pharm_pos', 'pharm_direction').

        Examples:
        - {{'shape', 'pharm'}}
            Surface + all pharmacophore sub-fields.
        - 'esp'
            Surface + ESP.
        - {{'pharm_positions', 'pharm_type'}}
            Inpaints positions and types, but not directions nor surface/ESP.

        The ``center_of_mass`` kwarg in the returned dict is always
        ``np.zeros(3)`` -- coordinate-frame handling for the condition
        (using it as given, centering on the scaffold/pharmacophore COM, or
        an explicit offset) is controlled by the ``condition_center_of_mass``
        argument of ``generate()`` / ``generate_composition()`` instead, not
        by this profile's ``com_before_centering``.
        """
        modalities = _resolve_modalities(condition_modalities) & self._present_modalities()
        ca = self.condition_atoms

        return {
            'surface': self.surface,
            'electrostatics': self.electrostatics,
            'pharm_types': self.pharm_types,
            'pharm_positions': self.pharm_positions,
            'pharm_directions': self.pharm_directions,
            'center_of_mass': np.zeros(3),
            'mol': self.mol,
            'atom_inds_to_inpaint': None if ca is None else ca.inds,
            'atom_types': None if ca is None else ca.types,
            'atom_pos': None if ca is None else ca.pos,
            'atom_formal_charges': None if ca is None else ca.formal_charges,
            'exit_vector_atom_inds': None if ca is None else ca.exit_vector_inds,
            **{flag: (name in modalities) for name, flag in _MODALITY_FLAGS.items()},
        }

    def to_dict(self) -> dict:
        """Serialize to a plain dict (numpy arrays + strings only)."""
        molblock = None
        if self.mol is not None:
            try:
                molblock = Chem.MolToMolBlock(self.mol)
            except Exception:
                pass
        return {
            'surface': self.surface,
            'electrostatics': self.electrostatics,
            'partial_charges': self.partial_charges,
            'pharm_types': self.pharm_types,
            'pharm_positions': self.pharm_positions,
            'pharm_directions': self.pharm_directions,
            'com_before_centering': self.com_before_centering,
            'smiles': self.smiles,
            'molblock': molblock,
            'n_atoms': self.n_atoms,
            'n_pharms': self.n_pharms,
            'condition_atoms': (
                None if self.condition_atoms is None
                else self.condition_atoms.to_dict()
            ),
            'neutralize_esp': self.neutralize_esp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'InteractionProfile':
        """Reconstruct from a dict produced by :meth:`to_dict`."""
        mol = None
        if d.get('molblock') is not None:
            try:
                mol = Chem.MolFromMolBlock(d['molblock'], removeHs=False)
            except Exception:
                pass
        if mol is None and d.get('smiles') is not None:
            try:
                mol = Chem.AddHs(Chem.MolFromSmiles(d['smiles']))
                from rdkit.Chem import AllChem
                AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
            except Exception:
                mol = None
        condition_atoms = None
        if d.get('condition_atoms') is not None:
            condition_atoms = ConditionAtoms.from_dict(d['condition_atoms'])
        elif d.get('atom_inds_to_inpaint'):
            condition_atoms = ConditionAtoms(
                inds=list(d['atom_inds_to_inpaint']),
                types=d.get('atom_types'),
                pos=None if d.get('atom_pos') is None else np.asarray(d['atom_pos']),
                formal_charges=(
                    None if d.get('atom_formal_charges') is None
                    else np.asarray(d['atom_formal_charges'])
                ),
                exit_vector_inds=d.get('exit_vector_atom_inds'),
            )
        if (
            condition_atoms is not None
            and mol is not None
            and condition_atoms.pos is None
        ):
            condition_atoms = ConditionAtoms.from_mol(
                mol,
                condition_atoms.inds,
                exit_vector_inds=condition_atoms.exit_vector_inds,
            )
        pharm_positions = d.get('pharm_positions', d.get('pharm_pos'))
        pharm_directions = d.get('pharm_directions', d.get('pharm_direction'))
        return cls(
            surface=None if d.get('surface') is None else np.asarray(d['surface']),
            electrostatics=None if d.get('electrostatics') is None else np.asarray(d['electrostatics']),
            partial_charges=None if d.get('partial_charges') is None else np.asarray(d['partial_charges']),
            pharm_types=None if d.get('pharm_types') is None else np.asarray(d['pharm_types']),
            pharm_positions=None if pharm_positions is None else np.asarray(pharm_positions),
            pharm_directions=None if pharm_directions is None else np.asarray(pharm_directions),
            com_before_centering=np.asarray(
                d.get('com_before_centering', d.get('center_of_mass', np.zeros(3)))
            ),
            smiles=d.get('smiles'),
            mol=mol,
            n_atoms=d.get('n_atoms') or (mol.GetNumAtoms() if mol is not None else 0),
            condition_atoms=condition_atoms,
            neutralize_esp=bool(d.get('neutralize_esp', False)),
        )

    @classmethod
    def from_generated_sample(
        cls,
        sample: dict | 'GeneratedSample',
        center: bool = True,
        xtb_optimize: bool = True,
        conversion: Literal['inferred', 'fixed'] = 'inferred',
        solvent: str = 'water',
        num_processes: int = 1,
        charge: int = 0,
        xtb_timeout: int = 60,
        neutralize_esp: bool = False,
    ) -> 'InteractionProfile' | None:
        """Build an :class:`InteractionProfile` from a ShEPhERD generated-sample
        dict (`GeneratedSample`).

        ``xtb_optimize`` selects whether to geometry-optimize with xtb before
        re-extracting surface / ESP / pharmacophores.

        ``conversion`` selects how molecular charge is obtained
        - ``'inferred'`` (default): :func:`~shepherd.extract.mol_charges_from_sample`
          — infers formal charge from xyz (trying 0, ±1, ±2).
        - ``'fixed'``: bond determination at a given charge (default 0)

        Arguments
        ----------
        sample : dict | GeneratedSample
            A generated sample.
        center : bool (default: True)
            Whether to translate the molecule's center of mass to the origin.
        xtb_optimize : bool (default: True)
            Whether to xtb-relax the generated geometry before re-extracting
            the profile.
        conversion : {'inferred', 'fixed'} (default: 'inferred')
            How molecular charge is obtained.
        solvent : str (default: 'water')
            Implicit solvent used by xTB.
        num_processes : int (default: 1)
            CPU cores used by xTB.
        charge : int (default: 0)
            Molecular charge for ``conversion='fixed'``.
        xtb_timeout : int (default: 60)
            xTB timeout in seconds for ``conversion='inferred'``.
            Increase for larger molecules.
        neutralize_esp : bool (default: False)
            If True, recompute ESP with net charge smeared over atoms
            (see :func:`compute_neutralized_esp`).

        Returns
        -------
        InteractionProfile, or None if no valid molecule can be extracted.
        """
        charges = None

        if conversion == 'inferred':
            from shepherd.extract import mol_charges_from_sample

            mol, charges = mol_charges_from_sample(
                sample,
                solvent=solvent,
                num_processes=num_processes,
                charge=None,
                xtb_optimize=xtb_optimize,
                xtb_timeout=xtb_timeout,
            )
        elif conversion == 'fixed':
            if xtb_optimize:
                from shepherd.extract import create_rdkit_molecule_xtb_optimized

                mol = create_rdkit_molecule_xtb_optimized(
                    sample,
                    solvent=solvent,
                    num_processes=num_processes,
                    charge=charge,
                )
            else:
                from shepherd.extract import create_rdkit_molecule

                mol = create_rdkit_molecule(sample, charge=charge)
        else:
            raise ValueError(
                f"conversion must be 'inferred' or 'fixed', got {conversion!r}"
            )

        if mol is None:
            return None
        return extract_interaction_profile(
            mol, xtb_optimize=False, partial_charges=charges, center=center,
            solvent=solvent, neutralize_esp=neutralize_esp,
        )

    def with_condition_atoms(
        self,
        inds: list[int],
        exit_vector_inds: list[int] | None = None,
    ) -> 'InteractionProfile':
        """Return a copy of ``self`` with ``condition_atoms`` set from ``self.mol`` at ``inds``.

        Use this to add scaffold-conditioning atoms to a profile that already
        carries surface / electrostatics / pharmacophore data (e.g. from
        :func:`extract_interaction_profile`), so x2/x3/x4 conditioning and
        scaffold conditioning apply together in ``generate()``.
        """
        if self.mol is None:
            raise ValueError("with_condition_atoms requires `self.mol` to be set.")
        return replace(
            self,
            condition_atoms=ConditionAtoms.from_mol(
                self.mol, inds, exit_vector_inds=exit_vector_inds
            ),
        )

    def translated(self, offset: np.ndarray) -> 'InteractionProfile':
        """Return a copy with ``surface`` and ``pharm_positions`` shifted by ``-offset``.
        """
        return replace(
            self,
            surface=None if self.surface is None else self.surface - offset,
            pharm_positions=None if self.pharm_positions is None else self.pharm_positions - offset,
        )

    def subselection(
        self,
        atom_inds: list[int],
        surface_atom_inds: list[int] | None = None,
        expand_pharm_consistent: bool = True,
        min_ring_priority_atoms: int = 1,
        restrict_esp: bool = True,
        probe_radius: float = 0.6,
        radial_buffer: float = 0.0,
        include_h: bool = False,
    ) -> 'InteractionProfile':
        """Return a copy of ``self`` with surface / electrostatics / pharmacophore
        data restricted to a chosen atom subset.

        Arguments
        ----------
        atom_inds : list[int]
            Seed atom indices into ``self.mol`` that drive the pharmacophore
            subselection. Also used for the surface/ESP restriction when
            ``surface_atom_inds`` is ``None``.
        surface_atom_inds : list[int] | None (default: None)
            If given, used in place of ``atom_inds`` to restrict the
            surface/ESP -- so ``atom_inds`` dictates which pharmacophores are
            kept while ``surface_atom_inds`` independently dictates which
            atoms the surface/ESP are regenerated over. Expanded via
            ``expand_pharm_consistent`` the same way ``atom_inds`` is
            (independently -- its expansion never affects which
            pharmacophores are kept).
        expand_pharm_consistent : bool (default: True)
            Whether to expand ``atom_inds`` (and, independently,
            ``surface_atom_inds`` when given) to the pharmacophore-consistent
            superset before restricting (see
            ``Pharmacophore.expand_atom_selection``).
        min_ring_priority_atoms : int (default: 1)
            Forwarded to the underlying pharmacophore expansion/filtering.
        restrict_esp : bool (default: True)
            If True, electrostatics are recomputed using only the selected
            atoms' charges. If False, ESP still reflects all atoms' charges,
            evaluated at the restricted surface.
        probe_radius : float (default: 0.6)
            Probe radius used to regenerate the restricted surface. Should
            match the value originally passed to `extract_interaction_profile`
            -- it isn't persisted on `InteractionProfile`, so pass it
            explicitly if it wasn't the default.
        radial_buffer : float (default: 0.0)
            Flat padding (in Angstroms) added to every retained atom's vdW
            radius before regenerating the surface.
        include_h : bool (default: False)
            If True, the directly-bonded hydrogen neighbors of every atom in
            the surface/ESP atom set are added before regenerating the
            surface/ESP.

        Returns
        -------
        InteractionProfile
        """
        if self.mol is None:
            raise ValueError("subselection requires `self.mol` to be set.")
        molec = Molecule(
            self.mol,
            probe_radius=probe_radius,
            surface_points=self.surface,
            partial_charges=self.partial_charges,
            electrostatics=self.electrostatics,
            pharm_multi_vector=None if self.pharm_types is None else False,
            pharm_types=self.pharm_types,
            pharm_ancs=self.pharm_positions,
            pharm_vecs=self.pharm_directions,
        )
        new_molec = molec.select_atoms(
            atom_inds,
            surface_atom_indices=surface_atom_inds,
            expand_pharm_consistent=expand_pharm_consistent,
            min_ring_priority_atoms=min_ring_priority_atoms,
            restrict_esp=restrict_esp,
            radial_buffer=radial_buffer,
            include_h=include_h,
        )
        return replace(
            self,
            surface=new_molec.surf_pos,
            electrostatics=new_molec.surf_esp,
            pharm_types=new_molec.pharm_types,
            pharm_positions=new_molec.pharm_ancs,
            pharm_directions=new_molec.pharm_vecs,
        )

    def pharm_prioritization_labels(
        self,
        atom_inds: list[int],
        min_ring_priority_atoms: int = 1,
    ):
        """Get the pharmacophore prioritization labels for a given set of atoms indices.
        For pharmacophore prioritization, ShepherdModel.generate() requires a list of ints
        with length equal to the number of pharmacophores in the profile, with 1 for high priority
        and 0 for low priority. This function sets pharmacophores associated (SMARTS patterns) with
        atoms in `atom_inds` as high priority, and the rest as low priority.

        Parameters
        ----------
        atom_inds : list[int]
            The indices of the atoms to get the pharmacophore prioritization labels for.
        min_ring_priority_atoms : int, optional
            For aromatic and ring-derived hydrophobe pharmacophores, the minimum
            number of selected ring heavy atoms (from ``atom_inds``) required to
            label the pharmacophore high priority. Default ``1``: any single
            selected ring atom is enough. Use a larger value (e.g. ``3``) when
            you want whole-ring pharmacophores only if several ring atoms were
            selected, not just one atom that happens to lie on a ring.

        Returns
        -------
        list[int]
            The pharmacophore prioritization labels for the given atoms where len == number of pharmacophores
            1: high priority (conditional), 0: low priority (inpainting)
            ex) [1, 1, 1, 0, 0]
        """
        molec = Molecule(
            self.mol,
            pharm_multi_vector=None if self.pharm_types is None else False,
            pharm_types=self.pharm_types,
            pharm_ancs=self.pharm_positions,
            pharm_vecs=self.pharm_directions,
        )
        pharm = molec.pharmacophore
        if pharm is not None and pharm.atom_ids is None:
            molec.get_pharmacophore(multi_vector=molec.pharm_multi_vector, return_atom_ids=True)
            pharm = molec.pharmacophore

        return pharm.priority_labels(atom_inds, min_ring_priority_atoms=min_ring_priority_atoms)


    def subselection_pharm(
        self,
        atom_inds: list[int] | None = None,
        min_ring_priority_atoms: int = 1,
        pharm_prioritization_labels: list[int] | None = None,
    ) -> 'InteractionProfile':
        """Return a copy of ``self`` with only the pharmacophores with high priority (1) kept.
        ``atom_inds`` and ``pharm_prioritization_labels`` are mutually exclusive.

        If ``atom_inds`` is provided, only the pharmacophores associated with the atoms in ``atom_inds`` are kept.
        If ``pharm_prioritization_labels`` is provided, only the pharmacophores with high priority (1) are kept.
        If both are provided, the pharmacophores with high priority (1) associated with the atoms in ``atom_inds`` are kept.
        """
        if atom_inds is not None and pharm_prioritization_labels is not None:
            raise ValueError("``atom_inds`` and ``pharm_prioritization_labels`` are mutually exclusive.")

        if atom_inds is not None:
            pharm_prioritization_labels = self.pharm_prioritization_labels(atom_inds, min_ring_priority_atoms=min_ring_priority_atoms)

        if pharm_prioritization_labels is not None:
            keep = np.asarray(pharm_prioritization_labels) == 1
        else:
            keep = np.ones(self.n_pharms, dtype=bool)

        return replace(
            self,
            pharm_types=None if self.pharm_types is None else self.pharm_types[keep],
            pharm_positions=None if self.pharm_positions is None else self.pharm_positions[keep],
            pharm_directions=None if self.pharm_directions is None else self.pharm_directions[keep],
        )

    @classmethod
    def from_condition_atoms(
        cls,
        mol: Chem.Mol,
        inds: list[int],
        exit_vector_inds: list[int] | None = None,
    ) -> 'InteractionProfile':
        """Build a profile with *only* a conditioned atom and no interaction profile.

        To scaffold-condition *and* condition on x2/x3/x4 together (the
        common case), call :func:`extract_interaction_profile` with
        ``condition_atom_inds=...`` instead, or attach atoms to an
        already-extracted profile via :meth:`InteractionProfile.with_condition_atoms`.

        ``mol`` is assumed to already be in the desired coordinate frame
        (typically centered). Types, positions, and formal charges are taken
        from ``mol`` at ``inds``.
        """
        try:
            smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))
        except Exception:
            smiles = None
        return cls(
            mol=mol,
            smiles=smiles,
            n_atoms=mol.GetNumAtoms(),
            condition_atoms=ConditionAtoms.from_mol(
                mol, inds, exit_vector_inds=exit_vector_inds
            ),
        )


def compute_neutralized_esp(molecule: Molecule) -> np.ndarray:
    """Recompute surface ESP after smearing net charge uniformly over atoms.

    Sets ``q_i -= Q/N`` so ``sum(q) == 0``, then evaluates Coulomb ESP on the
    same ``surf_pos``. Matches the neutral training distribution while preserving
    relative charge differences (dipole-and-higher). When formal charge is already
    0, returns ``molecule.surf_esp`` without recomputing.
    """
    if Chem.GetFormalCharge(molecule.mol) == 0:
        return molecule.surf_esp
    charges = np.asarray(molecule.partial_charges, dtype=np.float64)
    q_neutral = charges - charges.sum() / len(charges)
    centers = molecule.mol.GetConformer().GetPositions()
    dists = np.linalg.norm(molecule.surf_pos[:, np.newaxis] - centers, axis=2)
    inv = np.divide(1.0, dists, out=np.zeros_like(dists), where=dists > 0)
    esp_pa = (q_neutral @ inv.T) * COULOMB_SCALING
    esp_pa[~np.isfinite(esp_pa)] = 0.0
    return esp_pa.astype(molecule.surf_esp.dtype)


def extract_interaction_profile(
    mol: Chem.Mol,
    xtb_optimize: bool = True,
    partial_charges: np.ndarray | None = None,
    center: bool = True,
    solvent: str = 'water',
    condition_atom_inds: list[int] | None = None,
    exit_vector_atom_inds: list[int] | None = None,
    num_surf_points: int = 75,
    probe_radius: float = 0.6,
    neutralize_esp: bool = False,
) -> InteractionProfile | None:
    """Extract an :class:`InteractionProfile` from an RDKit molecule.

    Arguments
    ----------
    mol : Chem.Mol
        The molecule with a 3D conformer and explicit hydrogens.
    partial_charges : np.ndarray, optional
        Pre-computed partial charges (e.g. from xTB).  If ``None``,
        a single point xTB calculation is performed.
    center : bool
        Whether to translate the molecule so its center of mass is at the origin.
    solvent : str
        The solvent to use for the xTB partial charges. Default is 'water'.
    condition_atom_inds : list[int], optional
        Indices of the atoms to fix/inpaint; matches the atom order of the input ``mol``.
        If given, sets ``condition_atoms`` on the returned profile so the profile is
        ready for scaffold conditioning, if desired.
    exit_vector_atom_inds : list[int], optional
        Forwarded to :meth:`ConditionAtoms.from_mol` when ``condition_atom_inds``
        is given. Only applies to atom-inpainting.
    num_surf_points : int (default = 75)
        Number of surface points. ShEPhERD is trained only on 75 and cannot be changed
        for generation purposes.
        Evaluation uses ``num_surf_points=400``.
    probe_radius : float (default = 0.6)
        Probe radius (Å). ShEPhERD is trained only on 0.6, but this can be adjusted as desired.
        Evaluation uses ``probe_radius=1.2``.
    neutralize_esp : bool
        If True, replace raw surface ESP with
        :func:`compute_neutralized_esp` (no-op when formal charge is 0).
        Stored on the returned profile.

    Returns
    -------
    InteractionProfile
    """
    _partial_charges = None
    if xtb_optimize:
        mol, _, _partial_charges = optimize_conformer_with_xtb(
            mol, solvent=solvent,
            charge=Chem.GetFormalCharge(mol),
        )

    mol_coords = np.array(mol.GetConformer().GetPositions())
    com = np.mean(mol_coords, axis=0) if center else np.zeros(3)
    mol_coords_centered = mol_coords - com
    mol_centered = update_mol_coordinates(mol, mol_coords_centered)

    if partial_charges is None and _partial_charges is None:
        try:
            partial_charges = charges_from_single_point_conformer_with_xtb(
                mol_centered, charge=Chem.GetFormalCharge(mol_centered),
                solvent=solvent
            )
        except Exception:
            # Molecule container computes MMFF94 charges by default
            logging.warning("Failed to compute partial charges with xTB. Using MMFF94 charges.")

    elif _partial_charges is not None:
        partial_charges = _partial_charges

    try:
        molec = Molecule(
            mol_centered,
            num_surf_points=num_surf_points,
            probe_radius=probe_radius,
            partial_charges=partial_charges,
            pharm_multi_vector=False,
        )
    except Exception:
        return None

    try:
        smiles = Chem.MolToSmiles(Chem.RemoveHs(mol_centered))
    except Exception:
        smiles = None

    n_total = mol_centered.GetNumAtoms()  # includes H

    electrostatics = (
        compute_neutralized_esp(molec) if neutralize_esp else molec.surf_esp
    )
    profile = InteractionProfile(
        surface=molec.surf_pos,
        electrostatics=electrostatics,
        partial_charges=molec.partial_charges,
        pharm_types=molec.pharm_types,
        pharm_positions=molec.pharm_ancs,
        pharm_directions=molec.pharm_vecs,
        com_before_centering=com,
        smiles=smiles,
        mol=mol_centered,
        n_atoms=n_total,
        neutralize_esp=neutralize_esp,
    )
    if condition_atom_inds is not None:
        profile = profile.with_condition_atoms(
            condition_atom_inds, exit_vector_inds=exit_vector_atom_inds
        )
    return profile


_INPAINT_ADVANCED_OPTIONS_DEFAULTS = {
    'inpaint_x1_pos': False,
    'inpaint_x1_x': False,
    'inpaint_x1_bonds': False,
    'inpaint_x1_formal_charge': False,
    'stop_inpainting_at_time_x1_pos': 1.0,
    'stop_inpainting_at_time_x1_x': 1.0,
    'stop_inpainting_at_time_x1_bonds': 1.0,
    'inpaint_x3_pos': False,
    'inpaint_x3_x': False,
    'stop_inpainting_at_time_x3': 1.0,
    'add_noise_to_inpainted_x3_pos': 0.0,
    'add_noise_to_inpainted_x3_x': 0.0,
    'inpaint_x4_pos': False,
    'inpaint_x4_direction': False,
    'inpaint_x4_type': False,
    'stop_inpainting_at_time_x4': 1.0,
    'add_noise_to_inpainted_x4_pos': 0.0,
    'add_noise_to_inpainted_x4_direction': 0.0,
    'add_noise_to_inpainted_x4_type': 0.0,
}


@dataclass
class InpaintAdvancedOptions:
    """Extra inpainting options for :func:`shepherd.inference.sampler.generate`.

    inpaint_x{i}_{modality} : bool | None
        Whether to inpaint the {modality} of the {i}th representation.
    stop_inpainting_at_time_{i}_{modality} : float | None
        The time at which to stop inpainting the {modality} of the {i}th representation.
        0.0 is the noise prior, 1.0 is the data/end of denoising. For example,
        0.9 means inpaint 90% of the way through the denoising process.
    """
    inpaint_x1_pos: bool | None = None
    inpaint_x1_x: bool | None = None
    inpaint_x1_bonds: bool | None = None
    inpaint_x1_formal_charge: bool | None = None
    stop_inpainting_at_time_x1_pos: float | None = None
    stop_inpainting_at_time_x1_x: float | None = None
    stop_inpainting_at_time_x1_bonds: float | None = None

    inpaint_x3_pos: bool | None = None
    inpaint_x3_x: bool | None = None
    stop_inpainting_at_time_x3: float | None = None
    add_noise_to_inpainted_x3_pos: float | None = None
    add_noise_to_inpainted_x3_x: float | None = None

    inpaint_x4_pos: bool | None = None
    inpaint_x4_direction: bool | None = None
    inpaint_x4_type: bool | None = None
    stop_inpainting_at_time_x4: float | None = None
    add_noise_to_inpainted_x4_pos: float | None = None
    add_noise_to_inpainted_x4_direction: float | None = None
    add_noise_to_inpainted_x4_type: float | None = None

    def merge_overrides(self, other: 'InpaintAdvancedOptions') -> 'InpaintAdvancedOptions':
        """Return a copy of ``self`` with explicitly-set (non-``None``) fields from ``other`` applied."""
        kwargs = {}
        for f in fields(self):
            other_val = getattr(other, f.name)
            kwargs[f.name] = other_val if other_val is not None else getattr(self, f.name)
        return InpaintAdvancedOptions(**kwargs)

    def resolved(self) -> 'InpaintAdvancedOptions':
        """Return a copy with any remaining ``None`` fields filled with their hard defaults."""
        kwargs = {
            f.name: getattr(self, f.name) if getattr(self, f.name) is not None
            else _INPAINT_ADVANCED_OPTIONS_DEFAULTS[f.name]
            for f in fields(self)
        }
        return InpaintAdvancedOptions(**kwargs)
