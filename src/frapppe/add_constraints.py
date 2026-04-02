def reac2array(stoich_mat, reaction):
    return (stoich_mat.columns == reaction).astype(int)


def create_add_constraints_single(stoich_mat, fr_nom, fr_denom, a=0, b=1):
    set1 = set(fr_nom)
    set2 = set(fr_denom) - set1

    return np.array(
        [
            np.sum(
                [(1 - b) * stoich_mat.pipe(reac2array, r) for r in set1]
                + [-b * stoich_mat.pipe(reac2array, r) for r in set2],
                axis=0,
            ),
            np.sum(
                [(a - 1) * stoich_mat.pipe(reac2array, r) for r in set1]
                + [a * stoich_mat.pipe(reac2array, r) for r in set2],
                axis=0,
            ),
        ]
    )


def create_add_constraints(stoich_mat, bounds):
    mat_a = np.concatenate(
        [
            create_add_constraints_single(stoich_mat, *ratio, *ratio_bounds)
            for ratio_name, (ratio, ratio_bounds) in bounds.items()
        ],
        axis=0,
    )
    return mat_a
