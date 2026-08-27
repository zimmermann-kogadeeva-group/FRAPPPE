import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split

from .fluxes import get_flux_ratio


def train_rf(fluxes, mdvs, fr_nom, fr_denom, test_size=0.3, random_state=42):
    mdv_train, mdv_test, ratios_train, ratios_test = train_test_split(
        mdvs.transpose(),
        fluxes.pipe(get_flux_ratio, fr_nom, fr_denom).flux_ratio,
        test_size=test_size,
        random_state=random_state,
    )

    kfolds = KFold(n_splits=5, shuffle=True, random_state=random_state)
    rf = RandomForestRegressor(random_state=random_state)

    scores = cross_val_score(
        rf,
        mdv_train,
        ratios_train,
        cv=kfolds,
        scoring="neg_mean_absolute_error",
    )

    score_mean = np.mean(scores)
    score_sd = np.std(scores)

    rf.fit(mdv_train, ratios_train)

    ratios_pred = rf.predict(mdv_test)
    mae = mean_absolute_error(ratios_pred, ratios_test)
    scores = {"mae": mae, "cv_mae_mean": score_mean, "cv_mae_sd": score_sd}
    df_pred_sim = pd.DataFrame({"prediction": ratios_pred, "simulation": ratios_test})
    return rf, scores, df_pred_sim
