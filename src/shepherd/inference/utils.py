from numbers import Integral, Real


def resolve_early_stop_steps(early_stop_edm: int | float, num_steps: int) -> int:
    """Resolve ``early_stop_edm`` to the number of EDM steps to actually run.

    The full ``num_steps`` sigma schedule is still constructed; this only
    truncates the sampling loop.

    - ``-1``: no early stop (run all ``num_steps``)
    - ``int``: run this many steps (must be in ``[0, num_steps]``)
    - ``float`` in ``[0, 1]``: run ``int(early_stop_edm * num_steps)`` steps
    - ``float`` ``> 1``: run ``int(early_stop_edm)`` steps (absolute count)
    """
    if early_stop_edm < 0:
        return num_steps

    if isinstance(early_stop_edm, Integral):
        steps = int(early_stop_edm)
    elif isinstance(early_stop_edm, Real):
        val = float(early_stop_edm)
        if 0.0 <= val <= 1.0:
            steps = int(val * num_steps)
        elif val > 1.0:
            steps = int(val)
        else:
            raise ValueError(
                f"early_stop_edm as a float must be >= 0, got {early_stop_edm}"
            )
    else:
        raise TypeError(
            "early_stop_edm must be int or float, got "
            f"{type(early_stop_edm).__name__}"
        )

    if steps == 0:
        raise ValueError(
            "early_stop_edm resolved to 0 steps."
        )

    if not 0 <= steps <= num_steps:
        raise ValueError(
            f"early_stop_edm resolved to {steps} steps, which is outside "
            f"[0, {num_steps}]"
        )
    return steps


def _add_trajectories_to_generated_structures(
    generated_structures: list[dict],
    batch_size: int,
    trajectories: list[list[dict]],
    is_x0: bool = False
) -> None:
    """
    Helper function to add trajectory data to generated structures.

    Arguments
    ----------
    generated_structures : list[dict]
        The generated structures to add trajectories to.
    batch_size : int
        Number of batch elements.
    trajectories : list[list[dict]]
        List of trajectory frames, each containing batch data.
    is_x0 : bool, default=False
        Whether these are x0 prediction trajectories or regular trajectories.
    """
    trajectory_key = 'trajectories_x0' if is_x0 else 'trajectories'

    for b in range(batch_size):
        generated_structures[b][trajectory_key] = [
            {
                'x1': {
                    'atoms': traj[b]['x1']['atoms'],
                    'positions': traj[b]['x1']['positions'],
                    'bonds': traj[b]['x1']['bonds'],
                    'formal_charges': traj[b]['x1'].get('formal_charges', None),
                },
                'x2': {
                    'positions': traj[b]['x2']['positions'],
                },
                'x3': {
                    'charges': traj[b]['x3']['charges'],
                    'positions': traj[b]['x3']['positions'],
                },
                'x4': {
                    'types': traj[b]['x4']['types'],
                    'positions': traj[b]['x4']['positions'],
                    'directions': traj[b]['x4']['directions'],
                },
            }
            for traj in trajectories
        ]

