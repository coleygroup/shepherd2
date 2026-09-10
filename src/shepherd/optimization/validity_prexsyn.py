"""PrexSyn-based validity checker: projects SMILES into Enamine-compliant space.

Requires downloading the PrexSyn model: https://github.com/luost26/PrexSyn"""
from __future__ import annotations

from typing import Tuple

import torch
from rdkit import Chem

from prexsyn.shortcuts import AllInOneLoader, MoleculeProjector

from shepherd.optimization.individual import Individual
from shepherd.optimization.validity import ValidityChecker


class PrexSynProjectionValidityCheck(ValidityChecker):
    """Apply PrexSyn to project SMILES to Enamine-compliant space."""

    def __init__(self, config_path, num_samples: int = 16, threshold: float = 0.7):
        """Load the PrexSyn model and projector used to check validity.

        Arguments
        ---------
        config_path : str or Path
            Path to the PrexSyn model config.
        num_samples : int (default=16)
            Number of samples drawn per projection attempt.
        threshold : float (default=0.7)
            Minimum similarity for a projection to count as valid.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.threshold = threshold
        self.loader = AllInOneLoader(config_path)
        self.projector = MoleculeProjector(
            model=self.loader.model().to(self.device).eval(),
            detokenizer=self.loader.detokenizer(),
            descriptor="ecfp4",
            num_samples=num_samples,
        )

    def _neutralize_atoms(self, mol: Chem.Mol):
        """Neutralize formal charges on mol that have a compensating implicit H.

        Arguments
        ---------
        mol : Chem.Mol

        Returns
        -------
        Chem.Mol : mol, neutralized in place.
        """
        pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
        at_matches = mol.GetSubstructMatches(pattern)
        at_matches_list = [y[0] for y in at_matches]
        if len(at_matches_list) > 0:
            for at_idx in at_matches_list:
                atom = mol.GetAtomWithIdx(at_idx)
                chg = atom.GetFormalCharge()
                hcount = atom.GetTotalNumHs()
                atom.SetFormalCharge(0)
                atom.SetNumExplicitHs(hcount - chg)
                atom.UpdatePropertyCache()
        return mol

    def project(
        self, individual: Individual, n_tries: int = 5
    ) -> Tuple[str | None, float]:
        """Return the best PrexSyn synthesis projection and its similarity.

        Arguments
        ---------
        individual : Individual
        n_tries : int (default=5)
            Number of projection attempts to sample per target.

        Returns
        -------
        tuple[Optional[str], float] : best projected SMILES (or None) and its similarity.
        """
        mol = individual.mol
        if mol is None:
            mol = Chem.MolFromSmiles(individual.smiles)

        try:
            mol = self._neutralize_atoms(mol)
            smiles = Chem.MolToSmiles(mol)
            batch = self.projector.many([smiles] * n_tries)
        except Exception as e:
            print(f"Error occurred while projecting individual: {e}")
            return None, 0.0

        best_sim = 0.0
        best_syn_projection = None
        for target_result in batch.results:
            best = target_result.best()
            if best.similarity > best_sim:
                best_sim = best.similarity
                best_syn_projection = best.molecule.smiles()

        return best_syn_projection, best_sim

    def is_valid(self, individual: Individual) -> bool:
        """Project individual to Enamine-compliant space and accept it above threshold.

        Arguments
        ---------
        individual : Individual
            Updated in place with the projected SMILES/mol if accepted.

        Returns
        -------
        bool : True if the best projection's similarity meets self.threshold.
        """
        best_syn, best_sim = self.project(individual)
        if best_sim >= self.threshold:
            individual.smiles = best_syn
            individual.mol = Chem.MolFromSmiles(best_syn)
            return True
        return False
