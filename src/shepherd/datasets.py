# NOTE: open3d import moved to be lazy (inside functions) to avoid pickling issues
# with spawn multiprocessing. Open3D initializes thread locks that cannot be pickled.

from rdkit import Chem
import numpy as np
import torch
import torch_geometric

from shepherd.data_utils.subgraph import sample_random_or_subgraphs
from shepherd.diffusion.edm import EDMNoiseSchedule

def _lazy_import_shepherd_utils():
    """Lazy import of shepherd_score_utils to avoid open3d pickling issues with spawn multiprocessing."""
    import open3d  # noqa: F401 - must be imported before shepherd_score_utils
    from shepherd_score.generate_point_cloud import (
        get_atomic_vdw_radii,
        get_molecular_surface,
        get_electrostatics_given_point_charges,
    )
    from shepherd_score.pharm_utils.pharmacophore import get_pharmacophores, P_TYPES
    from shepherd_score.conformer_generation import update_mol_coordinates
    return get_atomic_vdw_radii, get_molecular_surface, get_electrostatics_given_point_charges, get_pharmacophores, update_mol_coordinates, P_TYPES

# Cache for shepherd_score lazy imports
_shepherd_utils_cache = {}

def _get_shepherd_utils():
    """Get cached shepherd utils, importing lazily on first use."""
    if not _shepherd_utils_cache:
        (
            _shepherd_utils_cache['get_atomic_vdw_radii'],
            _shepherd_utils_cache['get_molecular_surface'],
            _shepherd_utils_cache['get_electrostatics_given_point_charges'],
            _shepherd_utils_cache['get_pharmacophores'],
            _shepherd_utils_cache['update_mol_coordinates'],
            _shepherd_utils_cache['P_TYPES'],
        ) = _lazy_import_shepherd_utils()
    return _shepherd_utils_cache


def _separate_virtual_node_overlaps(
    pos: np.ndarray,
    virtual_node_mask: np.ndarray,
    min_distance: float = 1e-3,
) -> np.ndarray:
    """Move virtual nodes slightly if they exactly overlap non-virtual points."""
    if not virtual_node_mask.any() or (~virtual_node_mask).sum() == 0:
        return pos

    adjusted_pos = pos.copy()
    real_pos = adjusted_pos[~virtual_node_mask]
    offsets = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=adjusted_pos.dtype,
    )
    offsets = offsets / np.linalg.norm(offsets, axis=1, keepdims=True) * min_distance

    for virtual_node_idx in np.flatnonzero(virtual_node_mask):
        distances = np.linalg.norm(real_pos - adjusted_pos[virtual_node_idx], axis=1)
        if np.all(distances >= min_distance):
            continue

        for scale in (1.0, 2.0, 4.0, 8.0):
            for offset in offsets:
                candidate = adjusted_pos[virtual_node_idx] + offset * scale
                candidate_distances = np.linalg.norm(real_pos - candidate, axis=1)
                if np.all(candidate_distances >= min_distance):
                    adjusted_pos[virtual_node_idx] = candidate
                    break
            else:
                continue
            break

    return adjusted_pos


class HeteroDataset(torch_geometric.data.Dataset):
    def __init__(self,

            molblocks_and_charges,

            explicit_hydrogens = True,
            formal_charge_diffusion = False,

            x1 = True,
            x2 = True,
            x3 = True,
            x4 = True,

            recenter_x1 = True,
            add_virtual_node_x1 = True,
            remove_noise_COM_x1 = True,
            atom_types_x1 = [None, 'H', 'C', 'N', 'O', 'F', 'Cl', 'Br', 'I', 'S', 'P', 'Si'],
            charge_types_x1 = [0,1,2,-1,-2],
            bond_types_x1 = [None, 'SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'],
            scale_atom_features_x1 = 1.0,
            scale_bond_features_x1 = 1.0,

            recenter_x2 = False, # we want the center of x2 to be the virtual node (whose position is the center of x1)
            add_virtual_node_x2 = True,
            remove_noise_COM_x2 = False,
            num_points_x2 = 75,

            recenter_x3 = False,
            add_virtual_node_x3 = True,
            remove_noise_COM_x3 = False,
            num_points_x3 = 75,
            scale_node_features_x3 = 1.0,

            recenter_x4 = False,
            add_virtual_node_x4 = True, # must be true, for edge-case where molecule doesn't have any pharamcophores
            remove_noise_COM_x4 = False,
            max_node_types_x4 = 10, # number of pharmacophore types (can be set larger than represented in dataset)
            scale_node_features_x4 = 1.0,
            scale_vector_features_x4 = 1.0,
            multivectors = False,
            check_accessibility = False,

            probe_radius = 0.6,
            include_dummy_atoms = False,
            include_dummy_pharm = False,

            scale_point_cloud = 1.0,

            scaffold_conditioning = False,
            fixed_substructure_params_x1 = {},
            fixed_substructure_params_x4 = {},

            edm_schedule_dict = {},
        ):

        self.molblocks_and_charges = molblocks_and_charges
        self.length = len(molblocks_and_charges)

        self.edm_schedule_dict = edm_schedule_dict

        self.explicit_hydrogens = explicit_hydrogens
        assert self.explicit_hydrogens

        self.formal_charge_diffusion = formal_charge_diffusion

        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.x4 = x4

        self.recenter_x1 = recenter_x1
        self.add_virtual_node_x1 = add_virtual_node_x1
        self.remove_noise_COM_x1 = remove_noise_COM_x1 # True
        self.atom_types_x1 = atom_types_x1
        self.charge_types_x1 = charge_types_x1
        self.bond_types_x1 = bond_types_x1
        self.scale_atom_features_x1 = scale_atom_features_x1
        self.scale_bond_features_x1 = scale_bond_features_x1

        self.recenter_x2 = recenter_x2
        self.add_virtual_node_x2 = add_virtual_node_x2
        self.remove_noise_COM_x2 = remove_noise_COM_x2
        self.num_points_x2 = num_points_x2

        self.recenter_x3 = recenter_x3
        self.add_virtual_node_x3 = add_virtual_node_x3
        self.remove_noise_COM_x3 = remove_noise_COM_x3
        self.num_points_x3 = num_points_x3
        self.scale_node_features_x3 = scale_node_features_x3

        self.recenter_x4 = recenter_x4
        self.add_virtual_node_x4 = add_virtual_node_x4
        self.remove_noise_COM_x4 = remove_noise_COM_x4
        self.max_node_types_x4 = max_node_types_x4
        self.scale_node_features_x4 = scale_node_features_x4
        self.scale_vector_features_x4 = scale_vector_features_x4
        self.multivectors = multivectors
        self.check_accessibility = check_accessibility

        self.probe_radius = probe_radius
        self.scale_electrostatics = self.scale_node_features_x3  # alias

        self.include_dummy_atoms = include_dummy_atoms
        self.include_dummy_pharm = include_dummy_pharm

        self.scale_point_cloud = scale_point_cloud
        self.scaffold_conditioning = scaffold_conditioning
        self.fixed_substructure_params_x1 = fixed_substructure_params_x1
        self.fixed_substructure_params_x4 = fixed_substructure_params_x4

        self.edm_noise_schedule_pos = EDMNoiseSchedule(
            sigma_data=self.edm_schedule_dict.get('sigma_data_pos', 3.0),
            P_mean=self.edm_schedule_dict.get('P_mean', -1.2),
            P_std=self.edm_schedule_dict.get('P_std', 1.2),
        )
        self.edm_noise_schedule_atom_one_hot = EDMNoiseSchedule(
            sigma_data=self.edm_schedule_dict.get('sigma_data_atom_one_hot', 0.3),
            P_mean=self.edm_schedule_dict.get('P_mean', -1.2),
            P_std=self.edm_schedule_dict.get('P_std', 1.2),
        )
        self.edm_noise_schedule_bond = EDMNoiseSchedule(
            sigma_data=self.edm_schedule_dict.get('sigma_data_bond', 1.0),
            P_mean=self.edm_schedule_dict.get('P_mean', -1.2),
            P_std=self.edm_schedule_dict.get('P_std', 1.2),
        )
        if self.x3:
            self.edm_noise_schedule_esp = EDMNoiseSchedule(
                sigma_data=self.edm_schedule_dict.get('sigma_data_esp', 0.46),
                P_mean=self.edm_schedule_dict.get('P_mean', -1.2),
                P_std=self.edm_schedule_dict.get('P_std', 1.2),
            )
        if self.x4:
            self.edm_noise_schedule_pharm_one_hot = EDMNoiseSchedule(
                sigma_data=self.edm_schedule_dict.get('sigma_data_pharm_one_hot', 0.27),
                P_mean=self.edm_schedule_dict.get('P_mean', -1.2),
                P_std=self.edm_schedule_dict.get('P_std', 1.2),
            )
            self.edm_noise_schedule_direction = EDMNoiseSchedule(
                sigma_data=self.edm_schedule_dict.get('sigma_data_direction', 0.45),
                P_mean=self.edm_schedule_dict.get('P_mean', -1.2),
                P_std=self.edm_schedule_dict.get('P_std', 1.2),
            )


    def get_x1_data(
        self,
        mol,
        atom_inds_to_fix = [],
        scaffold_task = False,
        sigma: np.ndarray | None = None
    ):
        """
        Scaffold task is true if we are conditioning on a fixed substructure OR fixed pharmacophore (or both)
        """
        # this uses the same noise schedule for both positions and atom types/features

        if sigma is None:
            raise ValueError("sigma must be provided")

        data = {}
        data['sigma'] = torch.from_numpy(sigma.copy()).float()

        is_diffused_atom = np.ones(mol.GetNumAtoms(), dtype=bool)

        if self.include_dummy_atoms:
            num_dummy_atoms = np.random.randint(0, min(10, mol.GetNumAtoms()))
            is_diffused_atom = np.concatenate([is_diffused_atom, np.ones(num_dummy_atoms, dtype=bool)])
        else:
            num_dummy_atoms = 0

        # specify whether this training example is a scaffold task or not
        if len(atom_inds_to_fix) > 0:
            atom_inds_to_fix = np.array(atom_inds_to_fix) + int(self.add_virtual_node_x1) # accounting for virtual node
            scaffold_task = True

        atom_types = [self.atom_types_x1.index(a.GetSymbol()) for a in mol.GetAtoms()]
        if self.formal_charge_diffusion:
            formal_charges = [int(a.GetFormalCharge()) for a in mol.GetAtoms()]
            formal_charge_map = {c:self.charge_types_x1.index(c) for c in self.charge_types_x1}
            formal_charges_mapped = [formal_charge_map[f] for f in formal_charges]
        if num_dummy_atoms > 0:
            atom_types.extend([self.atom_types_x1.index('Dummy')] * num_dummy_atoms)
            if self.formal_charge_diffusion:
                formal_charges_mapped.extend([formal_charge_map[0]] * num_dummy_atoms)

        pos = np.array(mol.GetConformer().GetPositions())
        num_atoms = len(pos)

        bond_adj = 1-np.diag(np.ones(num_atoms + num_dummy_atoms, dtype = int))
        bond_adj = np.triu(bond_adj) # directed graph, to only include 1 edge per bond
        bond_edge_index = np.stack(bond_adj.nonzero(), axis = 0) # this doesn't include any edges to the virtual node
        bond_types_dict = {b:self.bond_types_x1.index(b) for b in self.bond_types_x1}
        max_bond_types_x1 = len(bond_types_dict)
        bond_types = []
        for b in range(bond_edge_index.shape[1]):
            idx_1 = int(bond_edge_index[0, b])
            idx_2 = int(bond_edge_index[1, b])
            if idx_1 >= num_atoms or idx_2 >= num_atoms:
                bond_types.append(bond_types_dict[None])
                continue
            bond = mol.GetBondBetweenAtoms(idx_1, idx_2)
            if bond is None:
                bond_types.append(bond_types_dict[None]) # non-bonded edge type; == 0
            else:
                bond_type = bond_types_dict[str(bond.GetBondType())]
                bond_types.append(bond_type)
        data['bond_edge_mask'] = torch.from_numpy((np.array(bond_types) != 0).copy()).bool() # True indicates a real bond

        if num_dummy_atoms > 0:
            if scaffold_task and len(atom_inds_to_fix) > 0: # don't sample atom_inds_to_fix if possible
                _true_atom_inds_to_fix = atom_inds_to_fix - int(self.add_virtual_node_x1)
                _atoms_to_sample = np.asarray(
                    [i for i in range(len(pos)) if i not in _true_atom_inds_to_fix],
                    dtype=np.int64,
                )
                _num_free_atoms = len(pos) - len(atom_inds_to_fix)
                if _num_free_atoms >= num_dummy_atoms:
                    dupl_atom_inds = np.random.choice(_atoms_to_sample, size = num_dummy_atoms, replace = False)
                else:
                    dupl_atom_inds = _atoms_to_sample
                    if num_dummy_atoms - len(_atoms_to_sample) > 0:
                        dupl_atom_inds = np.concatenate([
                            _atoms_to_sample,
                            np.random.choice(_true_atom_inds_to_fix, size = num_dummy_atoms - len(_atoms_to_sample), replace = False)
                            ])
            else:
                dupl_atom_inds = np.random.choice(num_atoms, size = num_dummy_atoms, replace = False)
            dupl_atom_inds = np.asarray(dupl_atom_inds, dtype=np.int64)
            dummy_atom_pos = pos[dupl_atom_inds] + np.random.randn(num_dummy_atoms, 3) * 0.2
            pos = np.concatenate([pos, dummy_atom_pos], axis = 0)

        pos = pos * self.scale_point_cloud

        COM_before_centering = pos.mean(0)[None, ...]
        data['com_before_centering'] = torch.from_numpy(COM_before_centering.copy()).float()
        pos_recentered = pos - pos.mean(0)
        # if scaffold task, we want to keep the molecule centered on the scaffold
        if self.recenter_x1 and not scaffold_task:
            pos = pos_recentered
        COM = pos.mean(0)[None, ...]
        data['com'] = torch.from_numpy(COM.copy()).float()

        virtual_node_mask = np.zeros(pos.shape[0] + int(self.add_virtual_node_x1))

        if self.add_virtual_node_x1: # should change according to desired behavior
            assert self.atom_types_x1[0] is None
            atom_types.insert(0, 0)
            bond_edge_index = bond_edge_index + 1 # accounting for virtual node
            if not scaffold_task:
                virtual_node_pos = COM # assumed that we already moved the molecule's COM to the origin
            else:
                # If scaffold task, we know that the scaffold is centered at the origin, so the virtual node should be at the origin
                virtual_node_pos = np.array([[0.0, 0.0, 0.0]])
            pos = np.concatenate([virtual_node_pos, pos], axis = 0) # setting virtual node position to (non-zero) COM
            pos_recentered = np.concatenate([virtual_node_pos * 0.0, pos_recentered], axis = 0) # setting virtual node position to zero
            virtual_node_mask[0] = 1
        virtual_node_mask = virtual_node_mask == 1
        num_nodes = num_atoms + int(self.add_virtual_node_x1) + num_dummy_atoms

        # Create true_atom_mask: True for real atoms (not virtual or dummy)
        true_atom_mask = np.zeros(num_nodes, dtype=bool)
        if self.add_virtual_node_x1:
            # Real atoms are indices 1 to num_atoms (virtual node is at index 0)
            true_atom_mask[1:num_atoms+1] = True
            is_diffused_atom = np.concatenate([np.zeros(1, dtype=bool), is_diffused_atom])
        else:
            # Real atoms are indices 0 to num_atoms-1 (no virtual node)
            true_atom_mask[:num_atoms] = True

        if scaffold_task and len(atom_inds_to_fix) > 0:
            is_diffused_atom[atom_inds_to_fix] = False # setting timestep to 0 for fixed substructure

        # Invariant: `~is_diffused_atom` must exactly match the set of indices held fixed
        # in the noising pipeline below (virtual node + atom_inds_to_fix). The model relies
        # on this correspondence (see model.py `is_scaffold = ~is_diffused_atom`).
        if __debug__:
            expected_fixed = np.zeros_like(is_diffused_atom)
            expected_fixed[virtual_node_mask] = True
            if scaffold_task and len(atom_inds_to_fix) > 0:
                expected_fixed[atom_inds_to_fix] = True
            assert np.array_equal(expected_fixed, ~is_diffused_atom), (
                "atom_inds_to_fix (plus virtual node) must exactly correspond to ~is_diffused_atom"
            )

        data['is_diffused_atom'] = torch.from_numpy(is_diffused_atom.copy()).bool()

        data['bond_edge_index'] = torch.from_numpy(bond_edge_index.copy()).long()
        data['pos'] = torch.from_numpy(pos.copy()).float()
        data['pos_recentered'] = torch.from_numpy(pos_recentered.copy()).float()
        data['virtual_node_mask'] = torch.from_numpy(virtual_node_mask.copy()).bool()


        # (scaled) one-hot embedding of atom types and formal charges for non-noised structure
        x = np.zeros((num_nodes, len(self.atom_types_x1))) #torch.tensor(atomic_numbers, dtype = torch.long)
        x[np.arange(num_nodes), atom_types] = 1
        x = x * self.scale_atom_features_x1
        if self.formal_charge_diffusion:
            x_formal_charges = np.zeros((len(formal_charges_mapped), len(self.charge_types_x1)))
            x_formal_charges[np.arange(len(formal_charges_mapped)), formal_charges_mapped] = 1
            x_formal_charges = x_formal_charges * self.scale_atom_features_x1
            if self.add_virtual_node_x1:
                # virtual node has all zeros for the formal charge one-hot features
                x_formal_charges = np.concatenate((np.zeros(len(self.charge_types_x1), dtype = x_formal_charges.dtype)[None, ...], x_formal_charges), axis = 0)
            x = np.concatenate((x, x_formal_charges), axis = 1)
        data['x'] = torch.from_numpy(x.copy()).float()


        # (scaled) one-hot embedding of bond types for non-noised structure
            # this doesn't include any edges to the virtual node
        bond_edge_x = np.zeros((bond_edge_index.shape[1], max_bond_types_x1))
        bond_edge_x[np.arange(len(bond_types)), bond_types] = 1
        bond_edge_x = bond_edge_x * self.scale_bond_features_x1
        data['bond_edge_x'] = torch.from_numpy(bond_edge_x.copy()).float()


        # forward noising non-virtual-nodes

        pos_noise = self.edm_noise_schedule_pos.get_noise(
            sigma,
            pos.shape,
            remove_COM_from_noise=self.remove_noise_COM_x1 and not scaffold_task,
            mask=virtual_node_mask,
            scale_by_sigma_data=self.edm_schedule_dict.get('scale_by_sigma_data', False),
        )

        x_noise = self.edm_noise_schedule_atom_one_hot.get_noise(
            sigma,
            x.shape,
            remove_COM_from_noise=False,
            mask=virtual_node_mask,
            scale_by_sigma_data=self.edm_schedule_dict.get('edm', {}).get('scale_by_sigma_data', False),
        )

        bond_edge_x_noise = self.edm_noise_schedule_bond.get_noise(
            sigma,
            bond_edge_x.shape,
            remove_COM_from_noise=False,
            mask=None,
            scale_by_sigma_data=self.edm_schedule_dict.get('edm', {}).get('scale_by_sigma_data', False),
        )

        if scaffold_task and len(atom_inds_to_fix) > 0:
            x_noise[atom_inds_to_fix] = 0.0
            pos_noise[atom_inds_to_fix] = 0.0

        data['pos_noise'] = torch.from_numpy(pos_noise.copy()).float()
        data['x_noise'] = torch.from_numpy(x_noise.copy()).float()
        data['bond_edge_x_noise'] = torch.from_numpy(bond_edge_x_noise.copy()).float()

        pos_forward_noised = pos + pos_noise
        x_forward_noised = x + x_noise
        bond_edge_x_forward_noised = bond_edge_x + bond_edge_x_noise

        if scaffold_task and len(atom_inds_to_fix) > 0:
            pos_forward_noised[atom_inds_to_fix] = pos[atom_inds_to_fix]
            x_forward_noised[atom_inds_to_fix] = x[atom_inds_to_fix]
        pos_forward_noised[virtual_node_mask] = pos[virtual_node_mask]
        x_forward_noised[virtual_node_mask] = x[virtual_node_mask]
        pos_forward_noised = _separate_virtual_node_overlaps(pos_forward_noised, virtual_node_mask)
        # TODO: add fixed bonds, too

        data['pos_forward_noised'] = torch.from_numpy(pos_forward_noised.copy()).float()
        data['x_forward_noised'] = torch.from_numpy(x_forward_noised.copy()).float()
        data['bond_edge_x_forward_noised'] = torch.from_numpy(bond_edge_x_forward_noised.copy()).float()

        return data, pos, virtual_node_mask, true_atom_mask



    def get_x2_data(
        self,
        radii,
        atom_centers,
        num_points,
        recenter,
        add_virtual_node,
        remove_noise_COM,
        virtual_node_pos = None,
        sigma: np.ndarray | None = None
    ):

        if sigma is None:
            raise ValueError("sigma must be provided")

        data = {}
        data['sigma'] = torch.from_numpy(sigma.copy()).float()

        utils = _get_shepherd_utils()
        pos = utils['get_molecular_surface'](
            atom_centers,
            radii,
            num_points=num_points,
            probe_radius = self.probe_radius,
            num_samples_per_atom = 20,
        )

        pos = pos * self.scale_point_cloud

        COM_before_centering = pos.mean(0)[None, :]
        data['com_before_centering'] = torch.from_numpy(COM_before_centering.copy()).float()
        pos_recentered = pos - pos.mean(0)
        if recenter:
            pos = pos_recentered
        COM = pos.mean(0)[None, :]
        data['com'] = torch.from_numpy(COM.copy()).float()

        virtual_node_mask = np.zeros(pos.shape[0] + int(add_virtual_node))
        if add_virtual_node: # should change according to desired behavior
            if (virtual_node_pos is None) or recenter:
                virtual_node_pos = COM
            pos = np.concatenate([virtual_node_pos, pos], axis = 0)
            pos_recentered = np.concatenate([virtual_node_pos * 0.0, pos_recentered], axis = 0)
            virtual_node_mask[0] = 1
        virtual_node_mask = virtual_node_mask == 1

        data['pos'] = torch.from_numpy(pos.copy()).float()
        data['pos_recentered'] = torch.from_numpy(pos_recentered.copy()).float()
        data['virtual_node_mask'] = torch.from_numpy(virtual_node_mask.copy()).bool()


        # one-hot embedding indicating real vs virtual nodes
        x = np.zeros((pos.shape[0], 2))
        x[~virtual_node_mask,0] = 1
        x[virtual_node_mask,1] = 1
        data['x'] = torch.from_numpy(x.copy()).float()
        data['x_forward_noised'] = data['x'] # there are no features to be noised in x2

        # forward noising non-virtual-nodes
        pos_noise = self.edm_noise_schedule_pos.get_noise(
            sigma,
            pos.shape,
            remove_COM_from_noise=remove_noise_COM,
            mask=virtual_node_mask,
            scale_by_sigma_data=self.edm_schedule_dict.get('scale_by_sigma_data', False),
        )
        pos_forward_noised = pos + pos_noise

        pos_forward_noised[virtual_node_mask] = pos[virtual_node_mask]
        pos_forward_noised = _separate_virtual_node_overlaps(pos_forward_noised, virtual_node_mask)

        data['pos_noise'] = torch.from_numpy(pos_noise.copy()).float()
        data['pos_forward_noised'] = torch.from_numpy(pos_forward_noised.copy()).float()

        return data, pos, virtual_node_mask



    def get_x3_data_electrostatics_only(
        self,
        charges,
        charge_centers,
        data,
        pos,
        virtual_node_mask,
        sigma: np.ndarray | None = None
    ):

        if sigma is None:
            raise ValueError("sigma must be provided")

        utils = _get_shepherd_utils()
        # x = utils['get_electrostatics_given_point_charges'](charges, charge_centers, pos) # compute ESP at each point in pos
        # Compute ESP only at non-virtual positions to avoid divide-by-zero when the virtual
        # node (e.g. at COM) coincides with a charge center. Virtual node values are set to 0 below.

        # rescale positions for ESP calculation
        pos_eval = pos[~virtual_node_mask] / self.scale_point_cloud if (self.scale_point_cloud != 1.0) else pos[~virtual_node_mask]
        x = np.zeros(pos.shape[0], dtype=np.float64)
        x[~virtual_node_mask] = utils['get_electrostatics_given_point_charges'](
            charges, charge_centers, pos_eval
        )
        x[virtual_node_mask] = 0.0
        x = x * self.scale_node_features_x3

        data['x'] = torch.from_numpy(x.copy()).float()

        x_noise = self.edm_noise_schedule_esp.get_noise(
            sigma,
            x.shape,
            remove_COM_from_noise=False,
            mask=virtual_node_mask,
            scale_by_sigma_data=self.edm_schedule_dict.get('scale_by_sigma_data', False),
        )
        x_forward_noised = x + x_noise

        x_forward_noised[virtual_node_mask] = x[virtual_node_mask]
        data['x_noise'] = torch.from_numpy(x_noise.copy()).float()
        data['x_forward_noised'] = torch.from_numpy(x_forward_noised.copy()).float()

        return data


    def get_x4_data(
        self,
        pharm_types,
        pos,
        direction,
        recenter,
        add_virtual_node,
        remove_noise_COM,
        virtual_node_pos = None,
        pharm_inds_to_fix = [],
        sigma: np.ndarray | None = None
    ):

        # it is  important to include a virtual node in case there are NO pharmacophores in the molecule
        assert add_virtual_node

        if sigma is None:
            raise ValueError("sigma must be provided")

        data = {}
        data['sigma'] = torch.from_numpy(sigma.copy()).float()

        utils = _get_shepherd_utils()

        pharm_types = pharm_types + 1 # need to accomodate potential virtual node as 0th index

        # add a small amount of noise to positions of pharmacophores to avoid identically overlapping points
        pos = pos + np.random.randn(*pos.shape) * 0.05

        non_aro_pharm_inds = np.where(pharm_types != utils['P_TYPES'].index('Aromatic') + 1)[0] # +1 for virtual node
        non_halo_pharm_inds = np.where(pharm_types != utils['P_TYPES'].index('Halogen') + 1)[0] # +1 for virtual node
        non_halo_and_aro_pharm_inds = np.array(list(set(non_halo_pharm_inds.tolist()).intersection(non_aro_pharm_inds.tolist())))

        if self.include_dummy_pharm and len(non_halo_and_aro_pharm_inds) > 0:
            # remove potential to sample aro or halogen pharmacophores since they overlap with
            # hydrophobes and doubles probability of selecting dummy pharmacophores
            n_dummy_pharms = np.random.randint(0, min(5, len(non_halo_and_aro_pharm_inds)))
            # dummy pharmacophore type is the last one (-1 since indexing starts at 0 then +1 for the virtual node)
            dummy_pharm_types = np.array([self.max_node_types_x4 - 1]) * np.ones(n_dummy_pharms, dtype = int)
            pharm_dummy_inds = np.random.choice(non_halo_and_aro_pharm_inds, size = n_dummy_pharms, replace = False)
            # pharm_dummy_inds = np.random.choice(pos.shape[0], size = n_dummy_pharms, replace = False)
            pharm_dummy_pos = pos[pharm_dummy_inds] + np.random.randn(n_dummy_pharms, 3) * 0.2
            pos = np.concatenate([pos, pharm_dummy_pos], axis = 0)
            pharm_types = np.concatenate([pharm_types, dummy_pharm_types])
            direction_dummy = np.random.randn(n_dummy_pharms, 3)
            direction_dummy /= np.linalg.norm(direction_dummy, axis=1, keepdims=True)
            direction_dummy[np.random.rand(n_dummy_pharms) > 0.5] = 0
            direction = np.concatenate([direction, direction_dummy], axis = 0)

        is_diffused_pharm = np.ones(pos.shape[0], dtype=bool)

        # no pharmacophores --> only virtual node remains
        if pharm_types.shape[0] == 0:

            pharm_types = np.array([0])
            is_diffused_pharm = np.zeros(1, dtype=bool)
            data['is_diffused_pharm'] = torch.from_numpy(is_diffused_pharm.copy()).bool()
            x = np.zeros((pharm_types.size, self.max_node_types_x4))
            x[np.arange(pharm_types.size), pharm_types] = 1
            x = x * self.scale_node_features_x4
            data['x'] = torch.from_numpy(x.copy()).float()

            if (virtual_node_pos is None) or recenter:
                virtual_node_pos = np.zeros(3)[None, ...]
            data['com_before_centering'] = torch.from_numpy(virtual_node_pos.copy()).float()
            data['com'] = torch.from_numpy(virtual_node_pos.copy()).float()

            virtual_node_mask = np.array([1])
            virtual_node_mask = virtual_node_mask == 1

            pos = virtual_node_pos
            direction = np.zeros(3)[None, ...]

            direction = direction * self.scale_vector_features_x4

            data['pos'] = torch.from_numpy(pos.copy()).float()
            data['pos_recentered'] = torch.from_numpy((pos * 0.0).copy()).float()
            data['direction'] = torch.from_numpy(direction.copy()).float()
            data['virtual_node_mask'] = torch.from_numpy(virtual_node_mask.copy()).bool()

            # virtual node remains unnoised
            x_noise = np.zeros(x.shape)
            data['x_noise'] = torch.from_numpy(x_noise.copy()).float()
            x_forward_noised = x
            data['x_forward_noised'] = torch.from_numpy(x_forward_noised.copy()).float()

            pos_noise = np.zeros(pos.shape)
            data['pos_noise'] = torch.from_numpy(pos_noise.copy()).float()
            pos_forward_noised = pos
            pos_forward_noised = _separate_virtual_node_overlaps(pos_forward_noised, virtual_node_mask)
            data['pos_forward_noised'] = torch.from_numpy(pos_forward_noised.copy()).float()

            direction_noise = np.zeros(direction.shape)
            data['direction_noise'] = torch.from_numpy(direction_noise.copy()).float()
            direction_forward_noised = direction
            data['direction_forward_noised'] = torch.from_numpy(direction_forward_noised.copy()).float()

            return data

        pos = pos * self.scale_point_cloud

        COM_before_centering = pos.mean(0)[None, :]
        data['com_before_centering'] = torch.from_numpy(COM_before_centering.copy()).float()
        pos_recentered = pos - pos.mean(0)
        if recenter:
            pos = pos_recentered
        COM = pos.mean(0)[None, :]
        data['com'] = torch.from_numpy(COM.copy()).float()


        virtual_node_mask = np.zeros(pos.shape[0] + int(add_virtual_node))
        if add_virtual_node: # should change according to desired behavior
            if (virtual_node_pos is None) or recenter:
                virtual_node_pos = COM

            pharm_types = np.concatenate([np.array([0]), pharm_types], axis = 0)
            pos = np.concatenate([virtual_node_pos, pos], axis = 0)
            pos_recentered = np.concatenate([virtual_node_pos * 0.0, pos_recentered], axis = 0)
            direction = np.concatenate([np.zeros(3)[None, ...], direction], axis = 0)

            virtual_node_mask[0] = 1
            is_diffused_pharm = np.concatenate([np.zeros(1, dtype=bool), is_diffused_pharm])
        virtual_node_mask = virtual_node_mask == 1

        # Offset pharm_inds_to_fix for virtual node
        if len(pharm_inds_to_fix) > 0:
            pharm_inds_to_fix = np.asarray(pharm_inds_to_fix) + int(add_virtual_node)

        if self.fixed_substructure_params_x4 and self.scaffold_conditioning:
            # NOTE: don't do anything fancy here since sigma is bounded for c_in
            # We mask it in model.py with `~is_diffused_pharm`
            is_diffused_pharm[pharm_inds_to_fix] = False # setting timestep to 0 for fixed substructure

        data['is_diffused_pharm'] = torch.from_numpy(is_diffused_pharm.copy()).bool()

        x = np.zeros((pharm_types.size, self.max_node_types_x4)) #torch.tensor(atomic_numbers, dtype = torch.long)
        x[np.arange(pharm_types.size), pharm_types] = 1
        x = x * self.scale_node_features_x4
        data['x'] = torch.from_numpy(x.copy()).float()

        data['pos'] = torch.from_numpy(pos.copy()).float()
        data['pos_recentered'] = torch.from_numpy(pos_recentered.copy()).float()

        direction = direction * self.scale_vector_features_x4
        data['direction'] = torch.from_numpy(direction.copy()).float()
        data['virtual_node_mask'] = torch.from_numpy(virtual_node_mask.copy()).bool()


        # forward noising non-virtual-nodes

        x_noise = self.edm_noise_schedule_pharm_one_hot.get_noise(
            sigma,
            x.shape,
            remove_COM_from_noise=False,
            mask=virtual_node_mask,
            scale_by_sigma_data=self.edm_schedule_dict.get('scale_by_sigma_data', False),
        )
        pos_noise = self.edm_noise_schedule_pos.get_noise(
            sigma,
            pos.shape,
            remove_COM_from_noise=remove_noise_COM and len(pharm_inds_to_fix) == 0,
            mask=virtual_node_mask,
            scale_by_sigma_data=self.edm_schedule_dict.get('scale_by_sigma_data', False),
        )
        direction_noise = self.edm_noise_schedule_direction.get_noise(
            sigma,
            direction.shape,
            remove_COM_from_noise=False,
            mask=virtual_node_mask,
            scale_by_sigma_data=self.edm_schedule_dict.get('scale_by_sigma_data', False),
        )

        if len(pharm_inds_to_fix) > 0:
            x_noise[pharm_inds_to_fix] = 0.0
            pos_noise[pharm_inds_to_fix] = 0.0
            direction_noise[pharm_inds_to_fix] = 0.0

        data['x_noise'] = torch.from_numpy(x_noise.copy()).float()
        data['pos_noise'] = torch.from_numpy(pos_noise.copy()).float()
        data['direction_noise'] = torch.from_numpy(direction_noise.copy()).float()

        x_forward_noised = x + x_noise
        pos_forward_noised = pos + pos_noise
        direction_forward_noised = direction + direction_noise

        x_forward_noised[virtual_node_mask] = x[virtual_node_mask]
        pos_forward_noised[virtual_node_mask] = pos[virtual_node_mask]
        direction_forward_noised[virtual_node_mask] = direction[virtual_node_mask]
        if len(pharm_inds_to_fix) > 0:
            x_forward_noised[pharm_inds_to_fix] = x[pharm_inds_to_fix]
            pos_forward_noised[pharm_inds_to_fix] = pos[pharm_inds_to_fix]
            direction_forward_noised[pharm_inds_to_fix] = direction[pharm_inds_to_fix]
        pos_forward_noised = _separate_virtual_node_overlaps(pos_forward_noised, virtual_node_mask)

        data['x_forward_noised'] = torch.from_numpy(x_forward_noised.copy()).float()
        data['pos_forward_noised'] = torch.from_numpy(pos_forward_noised.copy()).float()
        data['direction_forward_noised'] = torch.from_numpy(direction_forward_noised.copy()).float()

        return data


    def __getitem__(self, k):

        mol_block = self.molblocks_and_charges[k][0]
        charges = np.array(self.molblocks_and_charges[k][1]) # precomputed charges (e.g., from xTB)

        return self.process_item(mol_block, charges, k)


    def process_item(self, mol_block, charges, k):
        mol = Chem.MolFromMolBlock(mol_block, removeHs = False)

        # Lazy import to avoid pickling issues with spawn multiprocessing
        utils = _get_shepherd_utils()

        if self.x4:
            pharm_types, pharm_pos, pharm_direction = utils['get_pharmacophores'](
                mol,
                multi_vector = self.multivectors,
                check_access=self.check_accessibility,
            )

        scaffold_task = False
        atom_inds_to_fix = []
        pharm_inds_to_fix = []
        if self.scaffold_conditioning and (self.fixed_substructure_params_x1 or self.fixed_substructure_params_x4):
            if np.random.uniform(0, 1) < 0.5: # 50% chance to do "unconditional" task or scaffold task
                scaffold_task = True

                # If both x1 and x4 are available, decide which to fix: 33.3% x1 only, 33.3% x4 only, 33.3% both
                if self.fixed_substructure_params_x1 and self.fixed_substructure_params_x4:
                    _fix_mode = np.random.uniform(0, 1)
                    prob_fixed_pharmacophore = self.fixed_substructure_params_x4.get('prob_fixed_pharmacophore', 0.667)
                    _fix_x1_only = _fix_mode < (1.0 - prob_fixed_pharmacophore)
                    _fix_x4_only = (1.0 - prob_fixed_pharmacophore) <= _fix_mode < prob_fixed_pharmacophore
                    _fix_both = _fix_mode >= prob_fixed_pharmacophore
                else:
                    # If only one is available, always fix that one
                    _fix_x1_only = self.fixed_substructure_params_x1 and not self.fixed_substructure_params_x4
                    _fix_x4_only = self.fixed_substructure_params_x4 and not self.fixed_substructure_params_x1
                    _fix_both = False

                if self.fixed_substructure_params_x1 and self.x1 and (_fix_x1_only or _fix_both):
                    # doesn't account for virtual node
                    _fs_x1 = self.fixed_substructure_params_x1
                    _subgraph_kw = dict(
                        mol=mol,
                        prob_random_vs_subgraphs=_fs_x1.get('prob_random_vs_subgraphs', 0.5),
                        frac_random=_fs_x1.get('frac_random', 0.1),
                        hydrogen_weight=_fs_x1.get('hydrogen_weight', 0.3),
                        upper_bound_fraction_options=_fs_x1.get(
                            'upper_bound_fraction_options', [0.1, 0.2, 0.3]
                        ),
                    )
                    atom_inds_to_fix = sample_random_or_subgraphs(
                        **_subgraph_kw,
                        num_fracs=_fs_x1.get('num_fracs', 4),
                    )
                    _n_atoms = mol.GetNumAtoms()
                    if _n_atoms > 1 and len(atom_inds_to_fix) >= _n_atoms:
                        atom_inds_to_fix = sample_random_or_subgraphs(**_subgraph_kw, num_fracs=1)

                if self.fixed_substructure_params_x4 and self.x4 and len(pharm_types) > 0 and (_fix_x4_only or _fix_both):
                    n_pharm = len(pharm_types)
                    # randint(low, high) requires low < high (exclusive upper); n_pharm==1 ⇒ only choice
                    num_fixed_pharms = 1 if n_pharm == 1 else np.random.randint(1, n_pharm)
                    pharm_inds_to_fix = np.random.choice(n_pharm, size=num_fixed_pharms, replace=False)

            if len(atom_inds_to_fix) == 0 and len(pharm_inds_to_fix) == 0:
                scaffold_task = False

        # atomic_numbers = np.array([int(a.GetAtomicNum()) for a in mol.GetAtoms()])

        assert self.explicit_hydrogens # if we want to treat hydrogens implicitly, then we need to adjust how x2,x3,x4 are computed

        # centering molecule coordinates
        mol_coordinates = np.array(mol.GetConformer().GetPositions())
        orig_mol_coord_com = mol_coordinates.mean(0)
        orig_mol_coord_com_perturbed = orig_mol_coord_com + np.random.randn(3) * 0.15 # mostly within 0.5 angstroms
        if not scaffold_task:
            # center molcule COM at origin
            mol_coordinates = mol_coordinates - orig_mol_coord_com
            if self.x4:
                pharm_pos = pharm_pos - orig_mol_coord_com
            com = np.array([0.0, 0.0, 0.0])
        elif len(atom_inds_to_fix) > 0 and len(pharm_inds_to_fix) > 0:
            # Either use scaffold COM or original COM
            if np.random.uniform(0, 1) < self.fixed_substructure_params_x1.get('prob_use_scaffold_com', 1.0):
                remove_mean_com = np.concatenate([mol_coordinates[atom_inds_to_fix], pharm_pos[pharm_inds_to_fix]]).mean(0)
            else:
                remove_mean_com = orig_mol_coord_com_perturbed
            mol_coordinates = mol_coordinates - remove_mean_com
            pharm_pos = pharm_pos - remove_mean_com
            com = np.array([0.0, 0.0, 0.0])
        elif len(pharm_inds_to_fix) > 0:
            if np.random.uniform(0, 1) < self.fixed_substructure_params_x1.get('prob_use_scaffold_com', 1.0):
                remove_mean_com = pharm_pos[pharm_inds_to_fix].mean(0)
            else:
                remove_mean_com = orig_mol_coord_com_perturbed
            mol_coordinates = mol_coordinates - remove_mean_com
            pharm_pos = pharm_pos - remove_mean_com
            com = np.array([0.0, 0.0, 0.0])
        else:
            # center scaffold COM at origin
            if np.random.uniform(0, 1) < self.fixed_substructure_params_x1.get('prob_use_scaffold_com', 1.0):
                remove_mean_com = mol_coordinates[atom_inds_to_fix].mean(0)
            else:
                remove_mean_com = orig_mol_coord_com_perturbed
            mol_coordinates = mol_coordinates - remove_mean_com
            if self.x4:
                pharm_pos = pharm_pos - remove_mean_com
            com = np.array([0.0, 0.0, 0.0])
        #mol = update_mol_coordinates(mol, mol_coordinates, copy = False)
        mol = utils['update_mol_coordinates'](mol, mol_coordinates)

        radii = utils['get_atomic_vdw_radii'](mol)

        data_dict = {
            'molecule_id': torch.tensor([k], dtype=torch.long),
            'x1': {},
            'x2': {},
            'x3': {},
            'x4': {},
        }
        if self.scaffold_conditioning:
            data_dict['scaffold_task'] = torch.tensor([scaffold_task], dtype=torch.bool)

        sigma = self.edm_noise_schedule_pos.sample_sigma(1)

        if self.x1:
            x1_data, x1_pos, x1_virtual_node_mask, x1_true_atom_mask = self.get_x1_data(
                mol,
                atom_inds_to_fix=atom_inds_to_fix,
                scaffold_task=scaffold_task,
                sigma=sigma,
            )

            data_dict['x1'] = x1_data


        if self.x2:
            if self.x1:
                atom_centers = x1_pos[x1_true_atom_mask,:]
                atom_centers_with_dummy_atoms = x1_pos[~x1_virtual_node_mask,:]
                if not scaffold_task:
                    virtual_node_pos = atom_centers_with_dummy_atoms.mean(0)[None, ...] if (self.add_virtual_node_x2 and not self.recenter_x2) else None
                else:
                    virtual_node_pos = com[None, ...] if (self.add_virtual_node_x2 and not self.recenter_x2) else None
            else:
                atom_centers = mol_coordinates
                virtual_node_pos = None # this will get re-set to be the COM of x2 (NOT mol_coordinates) in get_x2_data

            # NOTE:
            # `x1_pos` has already been scaled by `scale_point_cloud` inside `get_x1_data`.
            # Since `get_x2_data` applies `pos = pos * self.scale_point_cloud`, we must pass *unscaled*
            # atom centers here to avoid scaling x2/x3 point clouds twice when x1 is enabled.
            atom_centers_surface = atom_centers / self.scale_point_cloud if (self.x1 and self.scale_point_cloud != 1.0) else atom_centers

            x2_data, x2_pos, x2_virtual_node_mask = self.get_x2_data(
                radii,
                atom_centers_surface,
                self.num_points_x2,
                self.recenter_x2,
                self.add_virtual_node_x2,
                self.remove_noise_COM_x2,
                virtual_node_pos = virtual_node_pos,
                sigma=sigma,
            )

            data_dict['x2'] = x2_data



        if self.x3:
            if self.x1:
                atom_centers = x1_pos[x1_true_atom_mask,:]
                atom_centers_with_dummy_atoms = x1_pos[~x1_virtual_node_mask,:]
                if not scaffold_task:
                    virtual_node_pos = atom_centers_with_dummy_atoms.mean(0)[None, ...] if (self.add_virtual_node_x3 and not self.recenter_x3) else None
                else:
                    virtual_node_pos = com[None, ...] if (self.add_virtual_node_x3 and not self.recenter_x3) else None
            else:
                atom_centers = mol_coordinates # this might need to be centered before we assign it to charge_centers
                virtual_node_pos = None # this will get re-set to be the COM of x3 (NOT mol_coordinates) in get_x3_data

            # See note in x2: avoid scaling the surface point cloud twice when x1 is enabled.
            atom_centers_surface = atom_centers / self.scale_point_cloud if (self.x1 and self.scale_point_cloud != 1.0) else atom_centers

            x3_data, x3_pos, x3_virtual_node_mask = self.get_x2_data(
                radii,
                atom_centers_surface,
                self.num_points_x3,
                self.recenter_x3,
                self.add_virtual_node_x3,
                self.remove_noise_COM_x3,
                virtual_node_pos = virtual_node_pos,
                sigma=sigma,
            )

            # the x3 point cloud, if re-centered, is displaced from the atom centers used to generate it.
                # Before computing electrostatics for x3, we have to displace the charge centers to account for this.
            x3_COM_displacement = (x3_data['com'] - x3_data['com_before_centering']).detach().cpu().numpy()
            charge_centers = atom_centers_surface + x3_COM_displacement

            # same noise is applied to both coordinates and features
            x3_data = self.get_x3_data_electrostatics_only(
                charges,
                charge_centers,
                x3_data,
                x3_pos,
                x3_virtual_node_mask,
                sigma=sigma,
            )

            data_dict['x3'] = x3_data


        if self.x4:

            if self.x1:
                atom_centers = x1_pos[x1_true_atom_mask,:]
                atom_centers_with_dummy_atoms = x1_pos[~x1_virtual_node_mask,:]
                if not scaffold_task:
                    virtual_node_pos = atom_centers_with_dummy_atoms.mean(0)[None, ...] if (self.add_virtual_node_x4 and not self.recenter_x4) else None
                else:
                    virtual_node_pos = com[None, ...] if (self.add_virtual_node_x4 and not self.recenter_x4) else None
            else:
                atom_centers = mol_coordinates
                virtual_node_pos = None # this will get re-set to be the COM of x4 (NOT mol_coordinates) in get_x4_data


            x4_data = self.get_x4_data(
                pharm_types, pharm_pos, pharm_direction,
                self.recenter_x4,
                self.add_virtual_node_x4,
                self.remove_noise_COM_x4 and not scaffold_task,
                virtual_node_pos = virtual_node_pos,
                pharm_inds_to_fix = pharm_inds_to_fix,
                sigma=sigma,
            )

            data_dict['x4'] = x4_data


        data = torch_geometric.data.HeteroData()
        if 'molecule_id' in data_dict:
            data.molecule_id = data_dict['molecule_id']

        if 'x1' in data_dict and data_dict['x1']:
            x1_data = data_dict['x1']

            x1_node_dict = {k: v for k, v in x1_data.items() if 'bond' not in k}

            x1_edge_dict = {
                'edge_index': x1_data['bond_edge_index'],
                'mask': x1_data['bond_edge_mask'],
                'x': x1_data['bond_edge_x'],
                'x_noise': x1_data['bond_edge_x_noise'],
                'x_forward_noised': x1_data['bond_edge_x_forward_noised'],
            }

            for key, value in x1_node_dict.items():
                data['x1'][key] = value
            for key, value in x1_edge_dict.items():
                data['x1', 'bond', 'x1'][key] = value

        if 'x2' in data_dict and data_dict['x2']:
            for key, value in data_dict['x2'].items():
                data['x2'][key] = value
        if 'x3' in data_dict and data_dict['x3']:
            for key, value in data_dict['x3'].items():
                data['x3'][key] = value
        if 'x4' in data_dict and data_dict['x4']:
            for key, value in data_dict['x4'].items():
                data['x4'][key] = value

        data['mol'] = mol
        if 'scaffold_task' in data_dict:
            data['scaffold_task'] = data_dict['scaffold_task']
        return data


    # for compatibility with other PyG versions
    def __len__(self): return self.length
    def len(self): return self.__len__()
    def getitem(self, k): return self.__getitem__(k)
    def get(self, k): return self.__getitem__(k)
