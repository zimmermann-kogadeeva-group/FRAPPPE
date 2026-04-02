import numpy as np
import pandas as pd


def get_reconex_fluxes(fluxes, factor=1, seed=42):
    rxs = fluxes.index

    reversible_mask = rxs.str.endswith(("_f", "_b"))
    reversible_rxs = rxs[reversible_mask].str.removesuffix(("_f", "_b")).unique()
    irreversible_rxs = rxs[~reversible_mask]

    net_fluxes = pd.DataFrame.from_dict(
        {r: fluxes.loc[r + "_f"] - fluxes.loc[r + "_b"] for r in reversible_rxs},
        orient="index",
    )

    rng = np.random.default_rng(seed)
    new_xchg = net_fluxes.apply(
        lambda x: factor * np.sign(x) * rng.uniform(0, np.abs(x)), axis=1
    )

    return pd.concat(
        [
            (net_fluxes + new_xchg)
            .where(net_fluxes > 0, -new_xchg)
            .rename(index=lambda x: x + "_f"),
            new_xchg.where(net_fluxes > 0, -(net_fluxes + new_xchg)).rename(
                index=lambda x: x + "_b"
            ),
            fluxes.loc[irreversible_rxs],
        ]
    ).reindex(rxs)


def get_xchg_fluxes(fluxes):
    rxs = fluxes.index

    reversible_mask = rxs.str.endswith(("_f", "_b"))
    reversible_rxs = rxs[reversible_mask].str.removesuffix(("_f", "_b")).unique()

    return pd.DataFrame.from_dict(
        {
            r: np.power(
                fluxes.loc[r + "_b"] / fluxes.loc[r + "_f"],
                np.sign(fluxes.loc[r + "_f"] - fluxes.loc[r + "_b"]),
            )
            for r in reversible_rxs
        },
        orient="index",
    )


def get_net_fluxes(fluxes, keep_irreversible=True):
    rxs = fluxes.index

    reversible_mask = rxs.str.endswith(("_f", "_b"))
    reversible_rxs = rxs[reversible_mask].str.removesuffix(("_f", "_b")).unique()
    irreversible_rxs = rxs[~reversible_mask]

    net_fluxes = pd.DataFrame.from_dict(
        {r: fluxes.loc[r + "_f"] - fluxes.loc[r + "_b"] for r in reversible_rxs},
        orient="index",
    )

    if keep_irreversible:
        return pd.concat([net_fluxes, fluxes.loc[irreversible_rxs]])
    else:
        return net_fluxes


# calculate flux ratios given a dictionary of flux ratios and the corresponding formulas
def get_flux_ratio(data, nominator, denominator, name="flux_ratio"):
    nom_fluxes = []
    denom_fluxes = []
    for r_id in nominator:
        if r_id + "_f" in data.index and r_id + "_b" in data.index:
            nom_fluxes.append(data.loc[r_id + "_f"] - data.loc[r_id + "_b"])
        else:
            nom_fluxes.append(data.loc[r_id])
    for r_id in denominator:
        if (
            r_id + "_f" in data.index and r_id + "_b" in data.index
        ):  # .str.startswith(r_id).sum() == 2:
            denom_fluxes.append(data.loc[r_id + "_f"] - data.loc[r_id + "_b"])
        else:
            denom_fluxes.append(data.loc[r_id])

    return pd.DataFrame(
        np.sum(nom_fluxes, axis=0) / np.sum(denom_fluxes, axis=0), columns=[name]
    )


def get_all_flux_ratios(fluxes, flux_ratios_dict):
    return pd.concat(
        [
            fluxes.pipe(get_flux_ratio, *reac_ratio)
            .assign(flux_ratio_name=name)
            .reset_index(names="iteration")
            for name, reac_ratio in flux_ratios_dict.items()
        ],
        ignore_index=True,
    )


def sample_if_enough(group, samples_per_bin, random_state=87):
    if len(group) >= samples_per_bin:
        return group.sample(samples_per_bin, replace=False, random_state=random_state)
    else:
        return group


def sample_flux_ratio_binned(
    fluxes,
    nominator,
    denominator,
    bins=None,
    samples_per_bin=250,
    random_state=87,
):
    bins = bins or np.linspace(0, 1, 11)
    return (
        get_flux_ratio(fluxes, nominator, denominator)
        .assign(
            binned=lambda x: pd.cut(
                x.flux_ratio, bins=bins, include_lowest=True, right=False
            )
        )
        .reset_index(names="iteration")
        .groupby("binned", as_index=False, group_keys=False)
        .apply(lambda group: sample_if_enough(group, samples_per_bin, random_state))
    )


def subset_fluxes_by_binned_flux_ratio(
    fluxes,
    nominator,
    denominator,
    bins=None,
    samples_per_bin=250,
    random_state=87,
):
    idxs = sample_flux_ratio_binned(
        fluxes, nominator, denominator, bins, samples_per_bin, random_state
    ).iteration
    return fluxes.iloc[:, idxs]


def subset_mdvs_by_binned_flux_ratio(
    mdvs,
    fluxes,
    nominator,
    denominator,
    bins=None,
    samples_per_bin=250,
    random_state=87,
):
    idxs = sample_flux_ratio_binned(
        fluxes, nominator, denominator, bins, samples_per_bin, random_state
    ).iteration
    return mdvs.iloc[:, idxs]


def sample_all_flux_ratio_binned(
    fluxes, flux_ratio_dict, bins=None, samples_per_bin=250, random_state=87
):
    return pd.concat(
        [
            sample_flux_ratio_binned(
                fluxes, *frac, bins, samples_per_bin, random_state
            ).assign(flux_ratio_name=ratio_name)
            for ratio_name, frac in flux_ratio_dict.items()
        ],
        ignore_index=True,
    )
