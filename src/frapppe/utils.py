import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def xy_line(axes, ls="--", color="black", **kwargs):
    x0, x1 = axes.get_xlim()
    y0, y1 = axes.get_ylim()
    axes.plot(*[[min(x0, y0), max(x1, y1)]] * 2, ls=ls, color=color, **kwargs)


def xy_line_plotly(figure, x0=0, y0=1, x1=1, y1=1, **kwargs):
    if "line" not in kwargs:
        kwargs["line"] = dict(dash="dash")
    if "row" not in kwargs:
        kwargs["row"] = "all"
    if "col" not in kwargs:
        kwargs["col"] = "all"
    figure.add_shape(type="line", x0=x0, y0=y0, x1=x1, y1=y1, **kwargs)


def melt_flux_df(data, **kwargs):
    return (
        data.reset_index(names="reaction_id")
        .melt(id_vars="reaction_id", var_name="sim_name", value_name="flux")
        .assign(**kwargs)
    )


def melt_mdv_df(data, **kwargs):
    return (
        data.melt(ignore_index=False, var_name="sim_name", value_name="mdv")
        .reset_index(names="emu")
        .pipe(
            lambda x: x.join(x.emu.str.extract(r"(?P<mb>.*)_(?P<atomnos>.*)_(?P<m>.*)"))
        )
        .assign(emu=lambda x: x.mb + "_" + x.atomnos)
    )


def plot_fluxes(fluxes, func=None, **kwargs):
    figsize = (10, 16) if "figsize" not in kwargs else kwargs.pop("figsize")
    func = func or sns.stripplot

    # If list is provided, convert to dict
    if isinstance(fluxes, (list, tuple)):
        fluxes = {i: x for i, x in enumerate(fluxes)}

    hue = None
    if isinstance(fluxes, dict):
        fluxes = pd.concat(
            [
                ind_flux_df.pipe(melt_flux_df, comparison=name)
                for name, ind_flux_df in fluxes.items()
            ]
        )
        hue = "comparison"
    elif isinstance(fluxes, pd.DataFrame):
        fluxes = fluxes.pipe(melt_flux_df)
    else:
        raise ValueError(f"Unrecognised type for fluxes argument: {type(fluxes)}")

    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(right=0.80)
    func(
        fluxes,
        y="reaction_id",
        x="flux",
        hue=hue,
        dodge=True,
    )
    if hue is not None:
        sns.move_legend(ax, loc="center left", bbox_to_anchor=(1.0, 0.5))
    return fig


def plot_mdvs_boxplots(data, col_order=None, **kwargs):
    fig = sns.FacetGrid(
        data.pipe(melt_mdv_df), col="emu", col_wrap=4, col_order=col_order
    ).map_dataframe(sns.boxplot, x="m", y="mdv", **kwargs)
    return fig


def plot_rf_results(
    results_dict, col_wrap=2, sharex=False, sharey=False, height=5, **kwargs
):
    df_pred_sim = pd.concat(
        [
            results[2].assign(flux_ratio_name=fr_name)
            for fr_name, results in results_dict.items()
        ],
        ignore_index=True,
    )

    fg = sns.FacetGrid(
        df_pred_sim,
        col="flux_ratio_name",
        col_wrap=col_wrap,
        sharex=sharex,
        sharey=sharey,
        height=height,
        **kwargs,
    ).map_dataframe(
        sns.scatterplot,
        x="simulation",
        y="prediction",
    )
    for fr_name, ax in fg.axes_dict.items():
        scores = results_dict[fr_name][1]
        ax.text(
            0.05,
            0.93,
            f"MAE: {scores['mae']:.3f}",
            fontsize=10,
            color="black",
            transform=ax.transAxes,
        )
        ax.text(
            0.05,
            0.88,
            f"CV MAE mean: {scores['cv_mae_mean']:.3f} \u00b1 {scores['cv_mae_sd']:.3f}",
            fontsize=10,
            color="black",
            transform=ax.transAxes,
        )
        xy_line(ax)

    return fg
