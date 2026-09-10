"""Genetic Algorithm operating in interaction space"""
from __future__ import annotations

import logging
import pickle
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import torch
import numpy as np
import pandas as pd
from rdkit import Chem

from shepherd.lightning_module import LightningModule
from shepherd.comp_inference.sampler import generate_composition
from shepherd.inference.sampler import generate

from shepherd.interaction_profile import (
    InteractionProfile,
    extract_interaction_profile,
)

from shepherd.generated_sample import GeneratedSample

from shepherd.optimization.individual import Individual
from shepherd.optimization.population import PopulationInitializer, SeedMoleculeInitializer
from shepherd.optimization.fitness import (
    FitnessOracle,
    DockingOracle,
)
from shepherd.optimization.validity import ValidityChecker, DrugLikenessChecker

torch.set_float32_matmul_precision('high')

logger = logging.getLogger(__name__)

@dataclass
class GAConfig:
    """All tuneable hyper-parameters for the abstract GA."""

    # --- population ---
    population_size: int = 20
    num_generations: int = 10
    skip_initial_eval: bool = False
    elite_fraction: float = 0.1
    max_iterations_fraction: float = 4.0

    # --- crossover ---
    crossover_prob: float = 0.3
    crossover_weight_a: float = 0.3
    crossover_weight_b: float = 0.3
    composition_mode: str = "default"

    # --- mutation ---
    mutation_mode: Literal["conditional", "composed", "ph4_conditioned"] = "conditional"
    mutation_rate: float = 1.0
    max_crossover_attempts: int = 10

    # --- selection ---
    selection_method: Literal["tournament", "pareto_tournament"] = "tournament"
    tournament_size: int = 16
    top_k: int | None = None

    # --- ShEPhERD-2 sampling ---
    N_x1_range: Tuple[int, int] = (-2, 2)
    N_x4_range: Tuple[int, int] = (0, 2)
    N_x4_max: int = 24
    batch_size: int = 4
    mutate_batch_size: int = 1
    num_steps: int = 400
    condition_mode: str = "all"

    # --- checkpointing ---
    checkpoint_dir: str | None = None
    checkpoint_every: int = 1

    # --- fragment merge ---
    fragment_merge_mode: bool = False
    fragment_atom_threshold: int = 20
    fragment_inclusion_prob: float = 0.1
    fragment_merge_alpha: float = 0.5
    fragment_selection_mode: Literal["lineage", "use_count"] = "lineage"

    # --- EDM sampler ---
    use_stochastic: bool = False
    shepherd_pred: bool = True
    early_stop_edm: int | float = 0.9
    sigma_max: float | None = 3
    sigma_min: float | None = 1e-3
    rho: float | None = 7.0
    alignment_start_frac: float = 0.0
    alignment_interval: int = 10
    alignment_mode: str = 'so3'
    alignment_ema_alpha: float = 1.0

    # --- molecule conversion ---
    xtb_optimize: bool = True
    profile_conversion: Literal['fixed', 'inferred'] = 'fixed'
    profile_solvent: str | None = 'water'

    # --- misc ---
    seed: int = 42
    verbose: bool = True

class InteractionSpaceGA:
    """Genetic Algorithm optimizer in interaction space with modular components"""

    def __init__(
        self,
        model_pl: LightningModule,
        oracle: FitnessOracle | None = None,
        config: GAConfig | None = None,
        validity_checker: ValidityChecker | None = None,
        population_initializer: PopulationInitializer | None = None,
        oracles: List[FitnessOracle] | None = None,
    ) -> None:
        """Wire up the GA's model, oracle(s), config, validity checker, and initializer.

        Arguments
        ---------
        model_pl : LightningModule
        oracle : Optional[FitnessOracle] (default=None)
            Single-objective oracle; mutually exclusive with oracles.
        config : Optional[GAConfig] (default=None)
        validity_checker : Optional[ValidityChecker] (default=None)
            Defaults to DrugLikenessChecker() if not provided.
        population_initializer : Optional[PopulationInitializer] (default=None)
            Defaults to SeedMoleculeInitializer(self.params) if not provided.
        oracles : Optional[List[FitnessOracle]] (default=None)
            Two or more oracles for multi-objective (Pareto) mode; mutually exclusive with oracle.
        """
        self.model = model_pl
        self.params = model_pl.params
        self.cfg = config or GAConfig()

        if oracles is not None:
            if len(oracles) < 2:
                raise ValueError(
                    "Pass at least 2 oracles for multi-objective mode, "
                    "or use the single 'oracle' parameter."
                )
            self.oracles: List[FitnessOracle] = list(oracles)
            self.oracle: FitnessOracle = oracles[0]  # primary oracle

            if self.cfg.selection_method == "tournament":
                self.cfg.selection_method = "pareto_tournament"
        elif oracle is not None:
            self.oracles = [oracle]
            self.oracle = oracle
        else:
            raise ValueError(
                "Provide either 'oracle' (single-objective) or 'oracles' (multi-objective)."
            )

        self.validity_checker: ValidityChecker = validity_checker or DrugLikenessChecker()
        self._initializer: PopulationInitializer = (
            population_initializer or SeedMoleculeInitializer(self.params)
        )
        self.rng = np.random.default_rng(self.cfg.seed)

        # Bookkeeping
        self.population: List[Individual] = []
        self.init_population: List[Individual] = []
        self.history: List[dict] = []
        self._start_generation: int = 0
        self._current_gen: int = 0
        self._fragment_use_counts: Dict[str, int] = {}
        self._lineage: Dict[str, Tuple[str, ...]] = {}
        self._all_time_best: List[Individual] = []


    def initialize_population(self, **kwargs) -> None:
        """Build the initial population via the configured initializer."""
        print("Initializing population ...")
        self.population = self._initializer.initialize(**kwargs)
        self.init_population = deepcopy(self.population)
        for ind in self.population:
            if ind.smiles:
                self._lineage[ind.smiles] = ind.parent_smiles
        if self.cfg.verbose:
            logger.info("Initialised population with %d individuals.", len(self.population))

    def run(self) -> List[Individual]:
        """Execute the GA loop."""
        assert len(self.population) > 0, (
            "Call initialize_population() before run()."
        )

        start_gen = self._start_generation

        if start_gen == 0:
            if not self.cfg.skip_initial_eval:
                self._evaluate_population(self.population)
                if self._is_multi_objective:
                    self._assign_pareto_ranks(self.population)
            if self.cfg.top_k is not None:
                self._update_all_time_best(self.population)
            self._record_generation(0)
            self._maybe_checkpoint(0)

        for gen in range(start_gen + 1, self.cfg.num_generations + 1):
            self._current_gen = gen
            t0 = time.time()
            if self.cfg.verbose:
                print(f"\n{'='*60}")
                print(f"  Generation {gen}/{self.cfg.num_generations}")
                print(f"{'='*60}")

            # 1. Elite selection
            n_elite = max(1, int(self.cfg.elite_fraction * self.cfg.population_size))
            if self._is_multi_objective:
                elites = self._select_pareto_elites(n_elite)
            else:
                sorted_pop = sorted(self.population, key=lambda ind: ind.fitness_score)
                elites = [deepcopy(ind) for ind in sorted_pop[:n_elite]]

            # 2. Produce offspring
            n_offspring_needed = self.cfg.population_size - n_elite
            if self.cfg.verbose:
                print(f"Producing {n_offspring_needed} offspring "
                      f"(elite fraction: {self.cfg.elite_fraction})...")
            offspring: List[Individual] = []

            crossover_attempts = 0
            max_iterations = n_offspring_needed * self.cfg.max_iterations_fraction
            iteration = 0

            while len(offspring) < n_offspring_needed and iteration < max_iterations:
                iteration += 1

                do_crossover = (
                    self.rng.random() < self.cfg.crossover_prob
                    and self.population
                    and len(self.population) > 1
                )
                if do_crossover:
                    parent_a, parent_b = self._select_parents()
                    new_individuals = self._crossover_generate(parent_a, parent_b, gen)

                    if not new_individuals:
                        crossover_attempts += 1
                        if crossover_attempts >= self.cfg.max_crossover_attempts:
                            logger.warning(
                                "AND-composition failed %d times; "
                                "falling back to single-parent mutation.",
                                crossover_attempts,
                            )
                            fallback_parent = min(
                                [parent_a, parent_b],
                                key=lambda p: p.fitness_score,
                            )
                            fallback = self._mutate_from_parent(
                                fallback_parent, gen,
                                parent_smiles=(parent_a.smiles, parent_b.smiles),
                            )
                            offspring.extend(self._filter_valid(fallback))
                            crossover_attempts = 0
                        continue

                    crossover_attempts = 0
                    if self.cfg.verbose:
                        print(f"  [crossover] produced {len(new_individuals)} candidates")
                else:
                    parent_a, parent_b = self._select_parents()
                    frag_smiles_set = (
                        {ind.smiles for ind in self.init_population}
                        if self.cfg.fragment_merge_mode and self.init_population
                        else set()
                    )

                    if frag_smiles_set and (
                        parent_a.smiles in frag_smiles_set
                        or parent_b.smiles in frag_smiles_set
                    ):
                        if self.cfg.verbose:
                            print(f"  [fragment->crossover] fragment parent forced to crossover: "
                                  f"A={parent_a.smiles} B={parent_b.smiles}")
                        new_individuals = self._crossover_generate(parent_a, parent_b, gen)

                    else:
                        if self.cfg.verbose:
                            print(f"  [mutation-only] parent: {parent_a.smiles} "
                                  f"(score={parent_a.fitness_score:.3f})")
                        new_individuals = self._mutate_from_parent(
                            parent_a, gen,
                            parent_smiles=(parent_a.smiles,),
                        )

                offspring.extend(self._filter_valid(new_individuals))

            if iteration >= max_iterations and len(offspring) < n_offspring_needed:
                logger.warning(
                    "Reached max iterations (%d) with only %d/%d offspring.",
                    max_iterations, len(offspring), n_offspring_needed,
                )

            offspring = offspring[:n_offspring_needed]

            # 3. Evaluate offspring
            self._evaluate_population(offspring)
            for ind in offspring:
                if ind.smiles:
                    self._lineage[ind.smiles] = ind.parent_smiles
            if self.cfg.top_k is not None:
                self._update_all_time_best(offspring)

            # 4. Build new population
            self.population = elites + offspring

            if self._is_multi_objective:
                self._assign_pareto_ranks(self.population)

            self._record_generation(gen)

            dt = time.time() - t0
            if self.cfg.top_k is not None:
                best_k = self.get_all_time_best()
                if self.cfg.verbose:
                    print(f"  Top-{self.cfg.top_k} best individuals:")
                    for i, ind in enumerate(best_k, start=1):
                        if self._is_multi_objective and ind.fitness_scores:
                            scores_str = ", ".join(f"{s:.4f}" for s in ind.fitness_scores)
                            print(
                                f"    {i:2d}. pareto_rank={ind.pareto_rank} | "
                                f"scores=[{scores_str}] | SMILES={ind.smiles}"
                            )
                        else:
                            print(
                                f"    {i:2d}. score={ind.fitness_score:.4f} | "
                                f"SMILES={ind.smiles}"
                            )
            else:
                best = min(self.population, key=lambda ind: ind.fitness_score)
                if self.cfg.verbose:
                    if self._is_multi_objective and best.fitness_scores:
                        scores_str = ", ".join(f"{s:.4f}" for s in best.fitness_scores)
                        print(f"  Best pareto rank: {best.pareto_rank} | scores: [{scores_str}]")
                    else:
                        print(f"  Best score: {best.fitness_score:.4f}")
                    print(f"  Best SMILES: {best.smiles}")
                    print(f"  Generation time: {dt:.1f}s")

            self._maybe_checkpoint(gen)

        if self._is_multi_objective:
            self.population.sort(key=lambda ind: (ind.pareto_rank, -ind.crowding_distance))
        else:
            self.population.sort(key=lambda ind: ind.fitness_score)
        return self.population

    def get_history_df(self):
        """Return per-generation statistics as a pandas DataFrame."""
        return pd.DataFrame(self.history)

    def get_best(self, n: int = 10) -> List[Individual]:
        """Return the top-n individuals ever evaluated."""
        if self._is_multi_objective:
            cache = self.oracle.score_cache
            pop_sorted = sorted(
                (ind for ind in self.population if ind.smiles is not None),
                key=lambda ind: (ind.pareto_rank, -ind.crowding_distance),
            )
            result = []
            seen: set = set()
            for ind in pop_sorted[:n]:
                copy = deepcopy(ind)
                copy.fitness_score = cache.get(ind.smiles, float(ind.pareto_rank))
                result.append(copy)
                seen.add(ind.smiles)
            if len(result) < n:
                for smi, score in sorted(cache.items(), key=lambda x: x[1]):
                    if smi not in seen:
                        mol = Chem.MolFromSmiles(smi)
                        # Look up per-oracle scores if available.
                        per_scores = [
                            o.score_cache.get(smi, float("inf")) for o in self.oracles
                        ]
                        result.append(Individual(
                            profile=InteractionProfile(
                                surface=np.zeros((1, 3)),
                                electrostatics=np.zeros(1),
                                pharm_types=np.zeros(0, dtype=int),
                                pharm_positions=np.zeros((0, 3)),
                                pharm_directions=np.zeros((0, 3)),
                            ),
                            smiles=smi,
                            mol=mol,
                            fitness_score=score,
                            fitness_scores=per_scores,
                            origin="unknown",
                        ))
                        seen.add(smi)
                    if len(result) >= n:
                        break
            return result
        else:
            cache = self.oracle.score_cache
            scored = sorted(cache.items(), key=lambda x: x[1])
            result = []
            for smi, score in scored[:n]:
                mol = Chem.MolFromSmiles(smi)
                result.append(Individual(
                    profile=InteractionProfile(
                        surface=np.zeros((1, 3)),
                        electrostatics=np.zeros(1),
                        pharm_types=np.zeros(0, dtype=int),
                        pharm_positions=np.zeros((0, 3)),
                        pharm_directions=np.zeros((0, 3)),
                    ),
                    smiles=smi,
                    mol=mol,
                    fitness_score=score,
                    origin="unknown",
                ))
            return result

    def get_all_time_best(self) -> List[Individual]:
        """Return all time best individuals (empty unless cfg.top_k is set)"""
        return list(self._all_time_best)

    def _filter_valid(self, individuals: List[Individual | None]) -> List[Individual]:
        """Keep only individuals that pass the validity checker."""
        result = []
        for ind in individuals:
            if ind is None:
                continue
            if ind.mol is None:
                continue
            if not self.validity_checker.is_valid(ind):
                continue
            result.append(ind)
        return result

    def _evaluate_population(self, population: List[Individual]) -> None:
        """Score every valid individual and update fitness score"""
        inds_to_eval: List[Individual] = []
        indices: List[int] = []

        for i, ind in enumerate(population):
            if ind.smiles is None:
                ind.fitness_score = float("inf")
                if self._is_multi_objective:
                    ind.fitness_scores = [float("inf")] * len(self.oracles)
            else:
                inds_to_eval.append(ind)
                indices.append(i)

        if not inds_to_eval:
            return

        if self._is_multi_objective:
            all_oracle_scores: List[List[float]] = []
            for oracle in self.oracles:
                all_oracle_scores.append(oracle.evaluate_with_cache(inds_to_eval))
                if oracle is self.oracle and isinstance(oracle, DockingOracle):
                    for pop_idx in indices:
                        self._update_from_docked_pose(population[pop_idx])
                elif (
                    oracle is self.oracle
                    and getattr(oracle, "docked_pose_buffer", None) is not None
                ):
                    for pop_idx in indices:
                        self._update_from_boltz_pose(population[pop_idx])

            for list_idx, pop_idx in enumerate(indices):
                scores = [
                    all_oracle_scores[o][list_idx] for o in range(len(self.oracles))
                ]
                population[pop_idx].fitness_scores = scores
                # Temporary scalar (sum); overwritten by _assign_pareto_ranks.
                population[pop_idx].fitness_score = float(sum(scores))
        else:
            scores = self.oracle.evaluate_with_cache(inds_to_eval)
            for idx, score in zip(indices, scores):
                population[idx].fitness_score = score
                if self.oracle is not None and isinstance(self.oracle, DockingOracle):
                    self._update_from_docked_pose(population[idx])
                elif (
                    self.oracle is not None
                    and getattr(self.oracle, "docked_pose_buffer", None) is not None
                ):
                    self._update_from_boltz_pose(population[idx])

    def _update_from_docked_pose(self, ind: Individual) -> None:
        """Re-extract profile from the docked conformer when available."""
        if ind.smiles is None:
            return
        pose_buffer = getattr(self.oracle, "docked_pose_buffer", None)
        if pose_buffer is None:
            return
        entry = pose_buffer.get(ind.smiles)
        if entry is None:
            return
        docked_mol = entry.get("docked_mol")
        if docked_mol is None:
            return
        new_profile = extract_interaction_profile(docked_mol, xtb_optimize=False)
        if new_profile is not None:
            ind.profile = new_profile
            ind.mol = new_profile.mol

    def _update_from_boltz_pose(self, ind: Individual) -> None:
        """Re-extract the interaction profile from the Boltz2-predicted pose."""
        if ind.smiles is None:
            return
        pose_buffer = getattr(self.oracle, "docked_pose_buffer", None)
        if pose_buffer is None:
            return
        entry = pose_buffer.get(ind.smiles)
        if entry is None:
            return
        boltz_mol = entry.get("docked_mol")
        if boltz_mol is None:
            return
        new_profile = extract_interaction_profile(boltz_mol, xtb_optimize=False)
        if new_profile is not None:
            ind.profile = new_profile
            ind.mol = new_profile.mol

    @property
    def _is_multi_objective(self) -> bool:
        """``True`` when more than one oracle is configured."""
        return len(self.oracles) > 1

    def _assign_pareto_ranks(self, population: List[Individual]) -> None:
        """Run NSGA-II non-dominated sorting on *population* in-place.

        Sets ``ind.pareto_rank`` (0 = Pareto front), ``ind.crowding_distance``,
        and updates ``ind.fitness_score = float(ind.pareto_rank)`` so that
        all scalar-based code paths keep working.

        Individuals with empty or missing ``fitness_scores`` are treated as
        having ``[inf, ...] x n_objectives``.
        """
        n = len(population)
        n_obj = len(self.oracles)

        obj_scores: List[List[float]] = [
            (ind.fitness_scores if len(ind.fitness_scores) == n_obj
             else [float("inf")] * n_obj)
            for ind in population
        ]

        # Non-dominated sorting (NSGA-II)
        domination_count = [0] * n       # how many individuals dominate i
        dominated_by_i: List[List[int]] = [[] for _ in range(n)]  # indices i dominates
        fronts: List[List[int]] = [[]]

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self._dominates(obj_scores[i], obj_scores[j]):
                    dominated_by_i[i].append(j)
                elif self._dominates(obj_scores[j], obj_scores[i]):
                    domination_count[i] += 1
            if domination_count[i] == 0:
                fronts[0].append(i)
                population[i].pareto_rank = 0

        front_idx = 0
        while fronts[front_idx]:
            next_front: List[int] = []
            for i in fronts[front_idx]:
                for j in dominated_by_i[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
                        population[j].pareto_rank = front_idx + 1
            front_idx += 1
            fronts.append(next_front)

        # Crowding distance
        for front in fronts[:-1]:  # last entry is always the empty sentinel
            if front:
                self._compute_crowding_distances(front, population, obj_scores)

        # Expose rank as the scalar fitness so sorting/tournament still work
        for ind in population:
            ind.fitness_score = float(ind.pareto_rank)

    @staticmethod
    def _dominates(scores_a: List[float], scores_b: List[float]) -> bool:
        """Return ``True`` if *a* Pareto-dominates *b*.

        *a* dominates *b* iff *a* is no worse on every objective and strictly
        better on at least one (lower scores = better for all objectives).
        """
        at_least_one_better = False
        for a, b in zip(scores_a, scores_b):
            if a > b:
                return False
            if a < b:
                at_least_one_better = True
        return at_least_one_better

    @staticmethod
    def _compute_crowding_distances(
        front_indices: List[int],
        population: List[Individual],
        obj_scores: List[List[float]],
    ) -> None:
        """Assign NSGA-II crowding distances to all individuals in a front."""
        n = len(front_indices)
        for i in front_indices:
            population[i].crowding_distance = 0.0

        if n <= 2:
            for i in front_indices:
                population[i].crowding_distance = float("inf")
            return

        n_obj = len(obj_scores[front_indices[0]])
        for obj in range(n_obj):
            sorted_front = sorted(front_indices, key=lambda i, obj=obj: obj_scores[i][obj])
            population[sorted_front[0]].crowding_distance = float("inf")
            population[sorted_front[-1]].crowding_distance = float("inf")

            obj_min = obj_scores[sorted_front[0]][obj]
            obj_max = obj_scores[sorted_front[-1]][obj]
            obj_range = obj_max - obj_min
            if obj_range == 0.0:
                continue

            for k in range(1, n - 1):
                population[sorted_front[k]].crowding_distance += (
                    (obj_scores[sorted_front[k + 1]][obj]
                     - obj_scores[sorted_front[k - 1]][obj])
                    / obj_range
                )

    def _select_pareto_elites(self, n: int) -> List[Individual]:
        """Return *n* deep-copied elites ordered by (pareto_rank, -crowding_distance)."""
        sorted_pop = sorted(
            self.population,
            key=lambda ind: (ind.pareto_rank, -ind.crowding_distance),
        )
        return [deepcopy(ind) for ind in sorted_pop[:n]]

    # internal: n_x1 / n_x4 sampling

    def _sample_N_x1(self, reference: int) -> int:
        """Sample the atom count to diffuse, offset from reference by cfg.N_x1_range."""
        lo, hi = self.cfg.N_x1_range
        return max(1, int(self.rng.integers(reference + lo, reference + hi + 1)))

    def _sample_N_x4(self, reference: int) -> int:
        """Sample the pharmacophore count to diffuse, capped at cfg.N_x4_max."""
        lo, hi = self.cfg.N_x4_range
        if reference < (self.cfg.N_x4_max - hi):
            return max(1, int(self.rng.integers(reference + lo, reference + hi + 1)))
        return reference

    # internal: selection

    def _get_used_fragment_smiles(self, individual: "Individual") -> set:
        """Return the set of init_population SMILES in individual's full ancestry.

        Uses self._lineage (smiles -> parent_smiles for every individual ever
        created) so ancestors replaced by selection are not missed.
        """
        frag_smiles = {ind.smiles for ind in self.init_population if ind.smiles}
        used: set = set()
        visited: set = set()
        queue = list(individual.parent_smiles)
        while queue:
            smi = queue.pop()
            if not smi or smi in visited:
                continue
            visited.add(smi)
            if smi in frag_smiles:
                used.add(smi)
            parents = self._lineage.get(smi)
            if parents:
                queue.extend(parents)
        return used

    def _select_parents(self) -> Tuple[Individual, Individual]:
        """Select two distinct parents with valid mols."""
        if self.cfg.selection_method == "pareto_tournament" or self._is_multi_objective:
            select_fn = self._pareto_tournament_select
        elif self.cfg.selection_method == "tournament":
            select_fn = self._tournament_select
        else:
            raise ValueError(f"Unknown selection method: {self.cfg.selection_method}")

        n_valid = sum(1 for ind in self.population if ind.mol is not None)

        parent_a = select_fn()
        for _ in range(50):
            if parent_a.mol is not None:
                break
            parent_a = select_fn()

        if n_valid < 2:
            return parent_a, parent_a

        parent_b = select_fn()
        for _ in range(50):
            if parent_b is not parent_a and parent_b.mol is not None:
                break
            parent_b = select_fn()

        if self.cfg.fragment_merge_mode and self.init_population:
            merge_now = (
                getattr(self, "_current_gen", 1) > 1
                and self.rng.random() < self.cfg.fragment_inclusion_prob
            )
            if merge_now:
                if self.cfg.fragment_selection_mode == "use_count":
                    counts = {ind.smiles: self._fragment_use_counts.get(ind.smiles, 0)
                              for ind in self.init_population}
                    min_count = min(counts.values())
                    pool = [ind for ind in self.init_population if counts[ind.smiles] == min_count]
                    unused_count = sum(1 for c in counts.values() if c == min_count)
                    print(
                        f"Merging with gen 0 fragment (unused={unused_count}/"
                        f"{len(self.init_population)}, mode=use_count)"
                    )
                    parent_b = pool[self.rng.integers(len(pool))]
                    self._fragment_use_counts[parent_b.smiles] = (
                        self._fragment_use_counts.get(parent_b.smiles, 0) + 1
                    )
                else:
                    used = (
                        self._get_used_fragment_smiles(parent_a)
                        | self._get_used_fragment_smiles(parent_b)
                    )
                    unused = [ind for ind in self.init_population if ind.smiles not in used]
                    pool = unused if unused else self.init_population
                    print(
                        f"Merging with gen 0 fragment (unused={len(unused)}/"
                        f"{len(self.init_population)}, mode=lineage)"
                    )
                    parent_b = pool[self.rng.integers(len(pool))]

        if self.cfg.verbose:
            if self._is_multi_objective:
                sa = (
                    ", ".join(f"{s:.3f}" for s in parent_a.fitness_scores)
                    if parent_a.fitness_scores else f"rank={parent_a.pareto_rank}"
                )
                sb = (
                    ", ".join(f"{s:.3f}" for s in parent_b.fitness_scores)
                    if parent_b.fitness_scores else f"rank={parent_b.pareto_rank}"
                )
                print(
                    f"  Selected parents: A([{sa}] rank={parent_a.pareto_rank}) | "
                    f"B([{sb}] rank={parent_b.pareto_rank})"
                )
            else:
                print(f"  Selected parents: "
                      f"A(score={parent_a.fitness_score:.3f}) | "
                      f"B(score={parent_b.fitness_score:.3f})")

        return parent_a, parent_b

    def _update_all_time_best(self, new_individuals: List[Individual]) -> None:
        """Merge new_individuals into the all-time-best archive and keep top-k."""
        candidates = [
            ind for ind in (self._all_time_best + new_individuals)
            if ind.mol is not None and np.isfinite(ind.fitness_score)
        ]
        # Deduplicate by SMILES, preserving first occurrence.
        seen: set = set()
        deduped: List[Individual] = []
        for ind in candidates:
            if ind.smiles not in seen:
                seen.add(ind.smiles)
                deduped.append(ind)

        if self._is_multi_objective:
            self._assign_pareto_ranks(deduped)
            deduped.sort(key=lambda ind: (ind.pareto_rank, -ind.crowding_distance))
        else:
            deduped.sort(key=lambda ind: ind.fitness_score)

        self._all_time_best = deduped[:self.cfg.top_k]

    def _get_parent_pool(self) -> List[Individual]:
        """Return the pool eligible for parent selection.

        When top_k is set and the archive is populated, returns the all-time
        top-k individuals across every generation evaluated so far.
        Otherwise falls back to the current population.
        """
        if self.cfg.top_k is not None and self._all_time_best:
            return self._all_time_best
        return self.population

    def _pareto_tournament_select(self) -> Individual:
        """NSGA-II binary tournament: lower pareto_rank wins, ties broken by crowding_distance."""
        pool = self._get_parent_pool()
        n = len(pool)
        candidate_idxs = self.rng.choice(n, size=min(self.cfg.tournament_size, n), replace=False)
        valid = [i for i in candidate_idxs if pool[i].mol is not None]
        idxs = valid if valid else list(candidate_idxs)
        best_idx = min(
            idxs,
            key=lambda i: (
                pool[i].pareto_rank,
                -pool[i].crowding_distance,
            ),
        )
        return pool[best_idx]

    def _tournament_select(self) -> Individual:
        """Pick the fittest of cfg.tournament_size random candidates from the parent pool."""
        pool = self._get_parent_pool()
        n = len(pool)
        candidate_idxs = self.rng.choice(n, size=min(self.cfg.tournament_size, n), replace=False)
        valid = [i for i in candidate_idxs if pool[i].mol is not None]
        idxs = valid if valid else list(candidate_idxs)
        best_idx = min(idxs, key=lambda i: pool[i].fitness_score)
        return pool[best_idx]

    # internal: crossover

    def _crossover_generate(
        self,
        parent_a: Individual,
        parent_b: Individual,
        generation: int,
    ) -> List[Individual]:
        """Produce offspring via AND-composition of two parent profiles."""
        prof_a, prof_b = parent_a.profile, parent_b.profile
        mol_a, mol_b = prof_a.mol, prof_b.mol

        if mol_a is None or mol_b is None:
            logger.warning("One or both parents lack a mol; using unaligned profiles.")
            cond_a, cond_b = prof_a, prof_b
        else:
            com_a = prof_a.com_before_centering
            com_b = prof_b.com_before_centering
            abs_ref_center = (com_a + com_b) / 2.0

            ref_center_a = abs_ref_center - com_a  # effective for already-centered mol_a
            ref_center_b = abs_ref_center - com_b  # effective for already-centered mol_b

            print(abs_ref_center)
            print(com_a, com_b)
            print("parent_a:", parent_a.smiles)
            print("parent_b:", parent_b.smiles)

            logger.debug(
                "Crossover: com_a=%s  com_b=%s  abs_ref=%s",
                np.round(com_a, 2), np.round(com_b, 2), np.round(abs_ref_center, 2),
            )

            cond_a = prof_a.translated(ref_center_a)
            cond_b = prof_b.translated(ref_center_b)

        weights = np.array([self.cfg.crossover_weight_a, self.cfg.crossover_weight_b])

        _is_fragment_merge = False
        _is_mixed_merge = False
        ha_a = ha_b = None
        if self.cfg.fragment_merge_mode and mol_a is not None and mol_b is not None:
            ha_a = mol_a.GetNumHeavyAtoms()
            ha_b = mol_b.GetNumHeavyAtoms()
            a_is_frag = ha_a <= self.cfg.fragment_atom_threshold
            b_is_frag = ha_b <= self.cfg.fragment_atom_threshold
            if a_is_frag and b_is_frag:
                _is_fragment_merge = True
            elif a_is_frag or b_is_frag:
                _is_mixed_merge = True

        if _is_fragment_merge:
            ref_n_atoms = prof_a.n_atoms + prof_b.n_atoms
            ref_n_pharms = prof_a.n_pharms + prof_b.n_pharms
        elif _is_mixed_merge:
            alpha = self.cfg.fragment_merge_alpha
            ref_n_atoms = (
                max(prof_a.n_atoms, prof_b.n_atoms)
                + int(alpha * min(prof_a.n_atoms, prof_b.n_atoms))
            )
            n_pharms_a, n_pharms_b = prof_a.n_pharms, prof_b.n_pharms
            ref_n_pharms = max(n_pharms_a, n_pharms_b) + int(alpha * min(n_pharms_a, n_pharms_b))
        else:
            ref_n_atoms = max(prof_a.n_atoms, prof_b.n_atoms)
            ref_n_pharms = max(prof_a.n_pharms, prof_b.n_pharms)

        N_x1 = self._sample_N_x1(ref_n_atoms)
        N_x4 = max(self._sample_N_x4(ref_n_pharms), ref_n_pharms)

        merge_note = ""
        if _is_fragment_merge:
            merge_note = f", fragment_merge=True [ha_a={ha_a}, ha_b={ha_b}]"
        elif _is_mixed_merge:
            merge_note = (
                f", mixed_merge=True alpha={self.cfg.fragment_merge_alpha} "
                f"[ha_a={ha_a}, ha_b={ha_b}]"
            )
        print(
            f"  Crossover N_x1: {N_x1}, N_x4: {N_x4} (ref atoms: {ref_n_atoms}, "
            f"ref pharma: {ref_n_pharms}{merge_note})"
        )

        generated = generate_composition(
            model_pl=self.model,
            N_x1=N_x1,
            N_x4=N_x4,
            batch_size=self.cfg.batch_size,
            profiles=[cond_a, cond_b],
            condition_modalities=self.cfg.condition_mode,
            weights_conditions=weights,
            composition_mode=self.cfg.composition_mode,
            num_steps=self.cfg.num_steps,
            use_stochastic=self.cfg.use_stochastic,
            shepherd_pred=self.cfg.shepherd_pred,
            early_stop_edm=self.cfg.early_stop_edm,
            sigma_max=self.cfg.sigma_max,
            sigma_min=self.cfg.sigma_min,
            rho=self.cfg.rho,
            alignment_start_frac=self.cfg.alignment_start_frac,
            alignment_interval=self.cfg.alignment_interval,
            alignment_mode=self.cfg.alignment_mode,
            alignment_ema_alpha=self.cfg.alignment_ema_alpha,
            store_trajectories=False,
            verbose=False,
        )

        torch.cuda.empty_cache()

        individuals = []
        for sample in generated:
            new_profile = self._profile_from_sample(sample)
            if new_profile is None:
                individuals.append(None)
            else:
                individuals.append(Individual(
                    profile=new_profile,
                    smiles=new_profile.smiles,
                    mol=new_profile.mol,
                    generation=generation,
                    parent_smiles=(parent_a.smiles or "", parent_b.smiles or ""),
                    origin="crossover",
                ))
        return individuals

    # internal: mutation

    def _mutate_from_parent(
        self,
        parent: Individual,
        generation: int,
        parent_smiles: Tuple[str, ...] = (),
    ) -> List[Individual]:
        """Single-parent mutation (fallback or mutation-only path)."""
        if self.cfg.mutation_mode == "composed":
            return self._mutate_composed(parent.profile, generation, parent_smiles)
        if self.cfg.mutation_mode == "ph4_conditioned":
            return self._mutate_ph4_conditioned(parent.profile, generation, parent_smiles)
        return self._mutate_conditional(parent.profile, generation, parent_smiles)

    def _mutate_conditional(
        self,
        profile: InteractionProfile,
        generation: int,
        parent_smiles: Tuple[str, ...] = (),
    ) -> List[Individual]:
        """Mode 1: plain conditional generation from a profile."""
        N_x1 = self._sample_N_x1(profile.n_atoms)
        N_x4 = self._sample_N_x4(profile.n_pharms)

        generated = generate(
            model_pl=self.model,
            batch_size=self.cfg.mutate_batch_size,
            N_x1=N_x1,
            N_x4=N_x4,
            condition=profile,
            condition_modalities=self.cfg.condition_mode,
            num_steps=self.cfg.num_steps,
            use_stochastic=self.cfg.use_stochastic,
            shepherd_pred=self.cfg.shepherd_pred,
            early_stop_edm=self.cfg.early_stop_edm,
            sigma_max=self.cfg.sigma_max,
            sigma_min=self.cfg.sigma_min,
            rho=self.cfg.rho,
            store_trajectories=False,
            verbose=False,
        )

        torch.cuda.empty_cache()

        individuals = []
        for sample in generated:
            new_profile = self._profile_from_sample(sample)
            if new_profile is None:
                individuals.append(None)
            else:
                individuals.append(Individual(
                    profile=new_profile,
                    smiles=new_profile.smiles,
                    mol=new_profile.mol,
                    generation=generation,
                    parent_smiles=parent_smiles,
                    origin="mutate_conditional",
                ))
        return individuals

    def _mutate_ph4_conditioned(
        self,
        profile: InteractionProfile,
        generation: int,
        parent_smiles: Tuple[str, ...] = (),
    ) -> List[Individual]:
        """Like _mutate_conditional but fixes all pharmacophores instead of inpainting."""
        N_x1 = self._sample_N_x1(profile.n_atoms)
        N_x4 = self._sample_N_x4(profile.n_pharms)

        generated = generate(
            model_pl=self.model,
            batch_size=self.cfg.mutate_batch_size,
            N_x1=N_x1,
            N_x4=N_x4,
            condition=profile,
            condition_modalities="all",
            pharmacophore_conditioning=True,
            num_steps=self.cfg.num_steps,
            use_stochastic=self.cfg.use_stochastic,
            shepherd_pred=self.cfg.shepherd_pred,
            early_stop_edm=self.cfg.early_stop_edm,
            sigma_max=self.cfg.sigma_max,
            sigma_min=self.cfg.sigma_min,
            rho=self.cfg.rho,
            store_trajectories=False,
            verbose=False,
        )

        torch.cuda.empty_cache()

        individuals = []
        for sample in generated:
            new_profile = self._profile_from_sample(sample)
            if new_profile is None:
                individuals.append(None)
            else:
                individuals.append(Individual(
                    profile=new_profile,
                    smiles=new_profile.smiles,
                    mol=new_profile.mol,
                    generation=generation,
                    parent_smiles=parent_smiles,
                    origin="mutate_ph4_conditioned",
                ))
        return individuals

    def _mutate_composed(
        self,
        profile: InteractionProfile,
        generation: int,
        parent_smiles: Tuple[str, ...] = (),
    ) -> List[Individual]:
        """Mode 2: single-condition AND-composition with unconditional."""
        weights = np.array([self.cfg.crossover_weight_a])

        N_x1 = self._sample_N_x1(profile.n_atoms)
        N_x4 = self._sample_N_x4(profile.n_pharms)

        generated = generate_composition(
            model_pl=self.model,
            N_x1=N_x1,
            N_x4=N_x4,
            batch_size=self.cfg.mutate_batch_size,
            profiles=[profile],
            condition_modalities=self.cfg.condition_mode,
            weights_conditions=weights,
            composition_mode="default",
            num_steps=self.cfg.num_steps,
            use_stochastic=self.cfg.use_stochastic,
            shepherd_pred=self.cfg.shepherd_pred,
            early_stop_edm=self.cfg.early_stop_edm,
            sigma_max=self.cfg.sigma_max,
            sigma_min=self.cfg.sigma_min,
            rho=self.cfg.rho,
            alignment_start_frac=self.cfg.alignment_start_frac,
            alignment_interval=self.cfg.alignment_interval,
            alignment_mode=self.cfg.alignment_mode,
            alignment_ema_alpha=self.cfg.alignment_ema_alpha,
            store_trajectories=False,
            verbose=False,
        )

        individuals = []
        for sample in generated:
            new_profile = self._profile_from_sample(sample)
            if new_profile is None:
                individuals.append(None)
            else:
                individuals.append(Individual(
                    profile=new_profile,
                    smiles=new_profile.smiles,
                    mol=new_profile.mol,
                    generation=generation,
                    parent_smiles=parent_smiles,
                    origin="mutate_composed",
                ))
        return individuals

    def _profile_from_sample(self, sample: GeneratedSample) -> InteractionProfile | None:
        """Build a profile from a generated sample.

        ``cfg.profile_conversion`` selects whether the molecular charge is
        pinned to 0 or inferred from the geometry; ``cfg.profile_solvent``
        selects the implicit solvent for the xTB calls (None = gas phase).
        """
        kwargs = dict(
            conversion=self.cfg.profile_conversion,
            xtb_optimize=self.cfg.xtb_optimize,
            solvent=self.cfg.profile_solvent,
        )
        if self.cfg.profile_conversion == "fixed":
            kwargs["charge"] = 0
        return InteractionProfile.from_generated_sample(sample, **kwargs)

    # internal: bookkeeping

    def _record_generation(self, gen: int) -> None:
        """Append a summary row for generation gen to self.history."""
        n_valid = sum(1 for ind in self.population if ind.is_valid)
        n_unique_evaluated = self.oracles[0].n_evaluated

        if self._is_multi_objective:
            best_ind = min(
                self.population,
                key=lambda i: (i.pareto_rank, -i.crowding_distance),
            )
            front0 = [ind for ind in self.population if ind.pareto_rank == 0]
            n_obj = len(self.oracles)
            # Per-objective stats across the Pareto front (rank-0 only)
            obj_best = [float("inf")] * n_obj
            obj_mean = [float("inf")] * n_obj
            if front0:
                for o in range(n_obj):
                    obj_vals = [
                        ind.fitness_scores[o] for ind in front0
                        if len(ind.fitness_scores) == n_obj
                        and np.isfinite(ind.fitness_scores[o])
                    ]
                    if obj_vals:
                        obj_best[o] = min(obj_vals)
                        obj_mean[o] = float(np.mean(obj_vals))
            stats = {
                "generation": gen,
                "n_valid": n_valid,
                "n_total": len(self.population),
                "pareto_front_size": len(front0),
                "best_pareto_rank": best_ind.pareto_rank,
                "best_scores": list(best_ind.fitness_scores),
                "obj_best": obj_best,
                "obj_mean": obj_mean,
                "best_smiles": best_ind.smiles,
                "best_origin": best_ind.origin,
                "best_parent_smiles": list(best_ind.parent_smiles),
                "n_unique_evaluated": n_unique_evaluated,
            }
            self.history.append(stats)
            if self.cfg.verbose:
                best_scores_str = (
                    ", ".join(f"{s:.4f}" for s in best_ind.fitness_scores)
                    if best_ind.fitness_scores else "n/a"
                )
                obj_best_str = ", ".join(f"{s:.4f}" for s in obj_best)
                print(f"  [Gen {gen}] valid={n_valid}/{len(self.population)}  "
                      f"front0={len(front0)}  best=[{best_scores_str}]  "
                      f"obj_best=[{obj_best_str}]  unique_evaluated={n_unique_evaluated}")
        else:
            scores = [
                ind.fitness_score for ind in self.population
                if np.isfinite(ind.fitness_score)
            ]
            best_ind = min(self.population, key=lambda i: i.fitness_score)
            stats = {
                "generation": gen,
                "n_valid": n_valid,
                "n_total": len(self.population),
                "best_score": min(scores) if scores else float("inf"),
                "mean_score": float(np.mean(scores)) if scores else float("inf"),
                "median_score": float(np.median(scores)) if scores else float("inf"),
                "std_score": float(np.std(scores)) if scores else float("inf"),
                "worst_score": max(scores) if scores else float("inf"),
                "best_smiles": best_ind.smiles,
                "best_origin": best_ind.origin,
                "best_parent_smiles": list(best_ind.parent_smiles),
                "n_unique_evaluated": n_unique_evaluated,
            }
            self.history.append(stats)
            if self.cfg.verbose:
                print(f"  [Gen {gen}] valid={n_valid}/{len(self.population)}  "
                      f"best={stats['best_score']:.4f}  mean={stats['mean_score']:.4f}  "
                      f"unique_evaluated={n_unique_evaluated}")

    def _maybe_checkpoint(self, gen: int) -> None:
        """Save a checkpoint if gen falls on the cfg.checkpoint_every cadence."""
        if self.cfg.checkpoint_dir is None:
            return
        if gen % self.cfg.checkpoint_every == 0:
            self.save_checkpoint(gen)

    def save_checkpoint(self, gen: int | None = None) -> Path:
        """Persist the full GA state to disk.

        Two files are written into ``checkpoint_dir``:

        * ``ga_checkpoint_gen{gen}.pkl``: the versioned snapshot.
        * ``ga_checkpoint_latest.pkl``  : always points to the newest
          snapshot so :meth:`resume` can find it automatically.

        Arguments
        ---------
        gen : int, optional
            Generation number used in the filename.  If ``None``, inferred
            from the last recorded history entry.

        Returns
        -------
        Path
            Path to the versioned checkpoint file that was written.
        """
        if self.cfg.checkpoint_dir is None:
            raise ValueError("checkpoint_dir is not set in GAConfig.")
        ckpt_dir = Path(self.cfg.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        if gen is None:
            gen = self.history[-1]["generation"] if self.history else 0

        state = {
            "generation": gen,
            "population": [ind.to_dict() for ind in self.population],
            "init_population": [ind.to_dict() for ind in self.init_population],
            "history": self.history,
            "all_evaluated": dict(self.oracle.score_cache),
            "all_evaluated_per_oracle": [dict(o.score_cache) for o in self.oracles],
            "rng_state": self.rng.bit_generator.state,
            "config": vars(self.cfg),
            "lineage": dict(self._lineage),
            "fragment_use_counts": dict(self._fragment_use_counts),
            "all_time_best": [ind.to_dict() for ind in self._all_time_best],
        }
        versioned = ckpt_dir / f"ga_checkpoint_gen{gen}.pkl"
        latest = ckpt_dir / "ga_checkpoint_latest.pkl"
        with open(versioned, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(latest, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        if self.cfg.verbose:
            print(f"  [Checkpoint] Saved generation {gen} -> {versioned}")
        return versioned

    def resume(self, checkpoint_path: str | None = None) -> int:
        """Restore GA state from a checkpoint and continue from next generation.

        Arguments
        ---------
        checkpoint_path : str, optional
            Explicit path to a ``.pkl`` checkpoint.  If ``None``,
            ``checkpoint_dir / ga_checkpoint_latest.pkl`` is used.

        Returns
        -------
        int
            The generation number that was restored.
        """
        if checkpoint_path is None:
            if self.cfg.checkpoint_dir is None:
                raise ValueError(
                    "No checkpoint_path given and checkpoint_dir is not set."
                )
            checkpoint_path = str(Path(self.cfg.checkpoint_dir) / "ga_checkpoint_latest.pkl")

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        with open(checkpoint_path, "rb") as f:
            state = pickle.load(f)

        self.population = [Individual.from_dict(d) for d in state["population"]]
        self.init_population = [Individual.from_dict(d) for d in state.get("init_population", [])]
        self.history = state["history"]
        per_oracle_caches = state.get("all_evaluated_per_oracle", [])
        for i, oracle in enumerate(self.oracles):
            if i < len(per_oracle_caches):
                oracle.score_cache.update(per_oracle_caches[i])
            elif i == 0:
                oracle.score_cache.update(state.get("all_evaluated", {}))
        self.rng.bit_generator.state = state["rng_state"]
        self._lineage.update(state.get("lineage", {}))
        self._fragment_use_counts.update(state.get("fragment_use_counts", {}))
        self._all_time_best = [
            Individual.from_dict(d) for d in state.get("all_time_best", [])
        ]
        if self.cfg.top_k is not None:
            self._update_all_time_best(self.population)
        restored_gen = state["generation"]
        self._start_generation = restored_gen

        if self.cfg.verbose:
            logger.info(
                "Resumed from checkpoint at generation %d (%d individuals, "
                "%d unique evaluated).",
                restored_gen, len(self.population), self.oracles[0].n_evaluated,
            )
        return restored_gen

    def save_results(self, path: str | None = None, n_best: int = 20) -> Path:
        """Write a ``ga_results.pkl`` summary file"""
        if path is None:
            if self.cfg.checkpoint_dir is None:
                raise ValueError(
                    "Provide a path or set checkpoint_dir in GAConfig."
                )
            path = str(Path(self.cfg.checkpoint_dir) / "ga_results.pkl")

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        all_evaluated: dict = dict(self.oracle.score_cache)
        all_evaluated_per_oracle: list = [dict(o.score_cache) for o in self.oracles]

        if self.cfg.top_k is not None:
            self._update_all_time_best(self.population)

        best_dicts = []
        if self._is_multi_objective:
            best_source = self._all_time_best if self.cfg.top_k is not None else self.population
            pop_smiles_seen: set = set()
            pop_sorted = sorted(
                (ind for ind in best_source if ind.smiles is not None),
                key=lambda ind: (ind.pareto_rank, -ind.crowding_distance),
            )
            for ind in pop_sorted[:n_best]:
                best_dicts.append(ind.to_dict())
                pop_smiles_seen.add(ind.smiles)
            if len(best_dicts) < n_best:
                for smi, score in sorted(all_evaluated.items(), key=lambda x: x[1]):
                    if smi not in pop_smiles_seen:
                        per_oracle_scores = [
                            cache.get(smi, float("inf"))
                            for cache in all_evaluated_per_oracle
                        ]
                        best_dicts.append({
                            "smiles": smi,
                            "fitness_score": score,
                            "fitness_scores": per_oracle_scores,
                            "pareto_rank": None,
                            "generation": None,
                            "parent_indices": [],
                            "parent_smiles": [],
                            "origin": "unknown",
                        })
                        pop_smiles_seen.add(smi)
                    if len(best_dicts) >= n_best:
                        break
        else:
            sorted_smiles = sorted(all_evaluated, key=lambda s: all_evaluated[s])
            for smi in sorted_smiles[:n_best]:
                score = all_evaluated[smi]
                match = next(
                    (ind for ind in self.population if ind.smiles == smi), None
                )
                if match is not None:
                    best_dicts.append(match.to_dict())
                else:
                    best_dicts.append({
                        "smiles": smi,
                        "fitness_score": score,
                        "generation": None,
                        "parent_indices": [],
                        "parent_smiles": [],
                        "origin": "unknown",
                    })

        results = {
            "final_population": [ind.to_dict() for ind in self.population],
            "best": best_dicts,
            "all_time_best": [ind.to_dict() for ind in self._all_time_best],
            "all_evaluated": all_evaluated,
            "oracle_cache": all_evaluated,              # alias expected by notebook
            "all_evaluated_per_oracle": all_evaluated_per_oracle,
            "history": self.history,
            "config": vars(self.cfg),
            "multi_objective": self._is_multi_objective,
        }

        with open(out_path, "wb") as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

        if self.cfg.verbose:
            print(f"  [Results] Saved to {out_path}")
        return out_path
