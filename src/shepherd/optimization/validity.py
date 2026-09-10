"""Validity checkers for filtering generated molecules."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple
import os
import sys

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, QED, RDConfig
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

from shepherd.optimization.individual import Individual


sa_path = os.path.join(RDConfig.RDContribDir, "SA_Score")
if sa_path not in sys.path:
    sys.path.append(sa_path)
import sascorer

class ValidityChecker(ABC):
    """Base class for molecule validity criteria evaluated on an :class:`Individual`."""

    @abstractmethod
    def is_valid(self, individual: Individual) -> bool:
        """Return ``True`` iff the molecule passes this validity criterion."""

    def _ensure_mol(self, individual: Individual) -> Chem.Mol | None:
        """Return individual.mol, parsing it from smiles if not already set.

        Returns
        -------
        Optional[Chem.Mol] : None if neither mol nor a parseable smiles is available.
        """
        if individual.mol is not None:
            return individual.mol
        if individual.smiles is None:
            return None
        return Chem.MolFromSmiles(individual.smiles)

    def __call__(self, individual: Individual) -> bool:
        """Alias for is_valid, so a checker instance can be used as a callable."""
        return self.is_valid(individual)


def _sa_score(mol: Chem.Mol) -> float | None:
    """Return RDKit's synthetic accessibility score for mol (lower = easier to make)."""
    return sascorer.calculateScore(mol)


@dataclass
class DrugLikenessChecker(ValidityChecker):
    """Enforce Lipinski's Rule of Five and optional stricter molecular quality filters."""

    mw_limit: float = 500.0
    logp_limit: float = 5.0
    hbd_limit: int = 5
    hba_limit: int = 10
    max_violations: int = 1

    # Optional extra descriptors
    tpsa_limit: float | None = None
    rotatable_bonds_limit: int | None = None
    num_rings_limit: int | None = None
    sa_score_limit: float | None = None
    qed_min: float | None = None
    require_parseable: bool = True
    verbose: bool = False

    def _extra_filter_checks(self, mol_noh: Chem.Mol) -> List[Tuple[bool, str, str]]:
        """Compute (exceeds, reject_message, info_message) for each configured optional filter."""
        checks: List[Tuple[bool, str, str]] = []

        if self.tpsa_limit is not None:
            tpsa = Descriptors.TPSA(mol_noh)
            checks.append((
                tpsa > self.tpsa_limit,
                f"  TPSA={tpsa:.1f} (limit {self.tpsa_limit}) exceeds limit",
                f"  TPSA={tpsa:.1f} (limit {self.tpsa_limit})",
            ))

        if self.rotatable_bonds_limit is not None:
            n_rot = rdMolDescriptors.CalcNumRotatableBonds(mol_noh)
            checks.append((
                n_rot > self.rotatable_bonds_limit,
                f"  Rotatable bonds={n_rot} (limit {self.rotatable_bonds_limit}) exceeds limit",
                f"  Rotatable bonds={n_rot} (limit {self.rotatable_bonds_limit})",
            ))

        if self.num_rings_limit is not None:
            n_rings = rdMolDescriptors.CalcNumRings(mol_noh)
            checks.append((
                n_rings > self.num_rings_limit,
                f"  Rings={n_rings} (limit {self.num_rings_limit}) exceeds limit",
                f"  Rings={n_rings} (limit {self.num_rings_limit})",
            ))

        if self.sa_score_limit is not None:
            sa = _sa_score(mol_noh)
            if sa is not None:
                checks.append((
                    sa > self.sa_score_limit,
                    f"  SA score={sa:.2f} (limit {self.sa_score_limit}) exceeds limit",
                    f"  SA score={sa:.2f} (limit {self.sa_score_limit})",
                ))

        if self.qed_min is not None:
            qed = QED.qed(mol_noh)
            checks.append((
                qed < self.qed_min,
                f"  QED={qed:.2f} (min {self.qed_min}) below minimum",
                f"  QED={qed:.2f} (min {self.qed_min})",
            ))

        return checks

    def is_valid(self, individual: Individual) -> bool:
        """Check individual against Lipinski's Rule of Five plus any configured extra filters."""
        mol = self._ensure_mol(individual)

        if mol is None:
            return not self.require_parseable

        mol_noh = Chem.RemoveHs(mol)
        mw = Descriptors.ExactMolWt(mol_noh)
        logp = Descriptors.MolLogP(mol_noh)
        hbd = rdMolDescriptors.CalcNumHBD(mol_noh)
        hba = rdMolDescriptors.CalcNumHBA(mol_noh)

        violations = sum(
            [
                mw > self.mw_limit,
                logp > self.logp_limit,
                hbd > self.hbd_limit,
                hba > self.hba_limit,
            ]
        )

        if violations > self.max_violations:
            return False

        extra_checks = self._extra_filter_checks(mol_noh)
        for exceeds, reject_message, _ in extra_checks:
            if exceeds:
                if self.verbose:
                    print(reject_message)
                return False

        if self.verbose:
            print(f"Valid: {individual.smiles}")
            print(
                f"  MW={mw:.1f} (limit {self.mw_limit}), "
                f"LogP={logp:.2f} (limit {self.logp_limit}), "
                f"HBD={hbd} (limit {self.hbd_limit}), "
                f"HBA={hba} (limit {self.hba_limit}), "
                f"violations={violations}"
            )
            for _, _, info_message in extra_checks:
                print(info_message)

        return True


@dataclass
class SubstructureFilter(ValidityChecker):
    """Reject molecules flagged by RDKit's curated substructure catalogues."""

    filter_pains: bool = True
    filter_brenk: bool = True
    custom_smarts: List[str] = field(default_factory=list)
    verbose: bool = False

    def __post_init__(self) -> None:
        """Build the PAINS/BRENK catalogue and compile any custom SMARTS filters."""
        params = FilterCatalogParams()
        if self.filter_pains:
            params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        if self.filter_brenk:
            params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
        self._catalog: object | None = (
            FilterCatalog(params) if (self.filter_pains or self.filter_brenk) else None
        )

        self._custom: List[Tuple[str, Chem.Mol]] = []
        for i, smarts in enumerate(self.custom_smarts):
            pat = Chem.MolFromSmarts(smarts)
            if pat is None:
                raise ValueError(f"Invalid custom SMARTS at index {i}: {smarts!r}")
            self._custom.append((smarts, pat))

    def is_valid(self, individual: Individual) -> bool:
        """Reject individual if it matches the PAINS/BRENK catalogue or a custom SMARTS."""
        mol = self._ensure_mol(individual)
        if mol is None:
            return False
        mol_noh = Chem.RemoveHs(mol)

        if self._catalog is not None:
            entry = self._catalog.GetFirstMatch(mol_noh)
            if entry is not None:
                if self.verbose:
                    print(f"  Rejected ({individual.smiles}): {entry.GetDescription()}")
                return False

        for smarts, pattern in self._custom:
            if mol_noh.HasSubstructMatch(pattern):
                if self.verbose:
                    print(f"  Rejected ({individual.smiles}): custom SMARTS '{smarts}'")
                return False

        if self.verbose:
            print(f"Valid with substructure filter: {individual.smiles}")

        return True

    def matched_patterns(self, individual: Individual) -> List[str]:
        """List descriptions of every catalogue/custom-SMARTS match for individual."""
        mol = self._ensure_mol(individual)
        if mol is None:
            return ["<unparseable>"]
        mol_noh = Chem.RemoveHs(mol)

        hits: List[str] = []
        if self._catalog is not None:
            for entry in self._catalog.GetMatches(mol_noh):
                hits.append(entry.GetDescription())

        for smarts, pattern in self._custom:
            if mol_noh.HasSubstructMatch(pattern):
                hits.append(f"custom:{smarts}")

        return hits


class CompositeValidityChecker(ValidityChecker):
    """Apply multiple :class:`ValidityChecker` instances with AND logic."""

    def __init__(self, checkers: List[ValidityChecker]) -> None:
        """Combine checkers, evaluated in order and short-circuiting on the first rejection."""
        self.checkers = checkers

    def is_valid(self, individual: Individual) -> bool:
        """Return True iff individual passes every checker in self.checkers."""
        if individual.mol is None and individual.smiles is not None:
            individual.mol = Chem.MolFromSmiles(individual.smiles)
        for checker in self.checkers:
            if not checker.is_valid(individual):
                return False
        return True

    def append(self, checker: ValidityChecker) -> None:
        """Add another checker to the AND-combined list."""
        self.checkers.append(checker)
