import multiprocessing as mp
from copy import deepcopy

import hopsy
import numpy as np
import pandas as pd
from freeflux import Model
from tqdm import tqdm

from .add_constraints import create_add_constraints
from .fluxes import get_reconex_fluxes


def _get_mdv_dict(mdv_res):
    return {
        f"{emu}_{idx}": x
        for emu in mdv_res.simulated_EMUs
        for idx, x in enumerate(mdv_res.simulated_MDV(emu).value)
    }


class MFA(object):
    def __init__(self, path_to_model, model_name="unnamed"):
        self.model = Model(model_name)
        self.model.read_from_file(path_to_model)
        self.stoich_matrix = self.model.get_total_stoichiometric_matrix()

    def make_bounds(self, lower, upper, **kwargs):
        assert lower <= upper
        lb = lower * np.ones(self.stoich_matrix.shape[1])
        ub = upper * np.ones(self.stoich_matrix.shape[1])
        if len(kwargs) > 0:
            for r_id, (lower, upper) in kwargs.items():
                r_idx = self.stoich_matrix.columns.get_loc(r_id)
                lb[r_idx] = lower
                ub[r_idx] = upper

        message = "All upper bounds should be greater than the lower bounds"
        assert np.all(lb <= ub), message

        return lb, ub

    def sample_fluxes(
        self,
        n_samples,
        bounds,
        *,
        thinning=10,
        seed=42,
        ss_exclude=None,
        add_constraints=None,
        add_constraints_b=None,
        reconex=None,
        **kwargs,
    ):
        # We get the vectors of upper and lower bounds
        lb, ub = self.make_bounds(**bounds)

        # Copied in case it is modified
        stoich_matrix = self.stoich_matrix.copy()

        A = np.concatenate(
            [-np.identity(stoich_matrix.shape[1]), np.identity(stoich_matrix.shape[1])]
        )
        b = np.concatenate([-lb, ub])

        if ss_exclude is not None:
            stoich_matrix = stoich_matrix[~stoich_matrix.index.isin(ss_exclude)]

        if add_constraints is not None:
            if isinstance(add_constraints, dict):
                add_constraints = create_add_constraints(
                    self.stoich_matrix, add_constraints
                )
            add_constraints_b = add_constraints_b or np.zeros(add_constraints.shape[0])
            A = np.concatenate([A, add_constraints])
            b = np.concatenate([b, add_constraints_b])

        problem = hopsy.add_equality_constraints(
            hopsy.Problem(A, b),
            A_eq=stoich_matrix.to_numpy(),
            b_eq=np.zeros(stoich_matrix.shape[0]),
        )

        starting_point = hopsy.compute_chebyshev_center(problem)

        chain = hopsy.MarkovChain(problem, starting_point=starting_point)
        rng = hopsy.RandomNumberGenerator(seed=seed)

        accrate, samples = hopsy.sample(
            chain, rng, n_samples=n_samples, thinning=thinning, **kwargs
        )

        samples = pd.DataFrame(
            samples[0].transpose(),
            index=stoich_matrix.columns,
        )
        if reconex is not None:
            samples = samples.pipe(get_reconex_fluxes, factor=reconex, seed=seed)

        return samples

    def get_single_sim_result(
        self, fluxes, target_emus, labeling_strategy, sim_name="sim", copy=True
    ):
        if copy:
            model = deepcopy(self.model)
        else:
            model = self.model

        sim = model.simulator("ss")

        sim.set_target_EMUs(target_emus)
        sim.set_labeling_strategy(**labeling_strategy)
        for r_name, flux in fluxes.items():
            sim.set_flux(r_name, flux)

        sim.prepare()
        res = sim.simulate()
        return res

    def simulate_mdvs(
        self,
        fluxes_df,
        target_emus,
        labeling_strategy,
        parallel=None,
        as_df=True,
        copy=True,
    ):
        if parallel is not None:
            vals = [
                (fluxes, target_emus, labeling_strategy, sim_name, copy)
                for sim_name, fluxes in fluxes_df.to_dict().items()
            ]
            pool = mp.Pool(parallel)
            all_mdv_res = pool.starmap(self.get_single_sim_result, vals)
        else:
            all_mdv_res = [
                self.get_single_sim_result(
                    fluxes, target_emus, labeling_strategy, sim_name, copy
                )
                for sim_name, fluxes in tqdm(fluxes_df.to_dict().items())
            ]

        df_freeflux_mdvs = pd.DataFrame(
            [_get_mdv_dict(x) for x in all_mdv_res],
        ).transpose()

        if as_df:
            return df_freeflux_mdvs
        else:
            return df_freeflux_mdvs.values
