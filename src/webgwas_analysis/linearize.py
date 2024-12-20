import polars as pl
import tqdm.auto as tqdm
from sklearn.linear_model import LinearRegression


def approximate_all(
    df: pl.DataFrame,
    endogs: list[str],
    exogs: list[str],
    eval_df: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Approximate all phenotypes in the dataframe using linear regression."""

    if eval_df is not None:
        missing_columns = set(exogs).difference(eval_df.columns)
        if len(missing_columns) > 0:
            raise ValueError(f"Missing columns in eval_df: {missing_columns}")

    # Check for missingness
    phenotypes_with_missingness = (
        df.select(endogs + exogs)
        .select(pl.all().is_null())
        .unpivot()
        .filter(pl.col("value").gt(0))["variable"]
        .to_list()
    )
    if len(phenotypes_with_missingness) > 0:
        return approximate_all_with_missingness(df, endogs, exogs, eval_df)

    return _approximate(df, endogs, exogs, eval_df)


def approximate_all_with_missingness(
    df: pl.DataFrame,
    endogs: list[str],
    exogs: list[str],
    eval_df: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Approximate all phenotypes in the dataframe using linear regression when
    missingness is present. Note, this is considerably slower than `approximate_all`.
    """

    if eval_df is None:
        eval_df = df
    else:
        # Check that eval_df has a column for every exog
        missing_columns = set(exogs).difference(eval_df.columns)
        if len(missing_columns) > 0:
            raise ValueError(f"Missing columns in eval_df: {missing_columns}")

    approx_df = dict()
    for endog in tqdm.tqdm(endogs):
        this_df = df.select([endog] + exogs).drop_nulls()
        Yhat_df = _approximate(this_df, endog, exogs, eval_df)
        assert Yhat_df.shape[1] == 1
        approx_df[endog] = Yhat_df[endog]

    return pl.DataFrame(approx_df)


def _approximate(
    df: pl.DataFrame,
    endog: str | list[str],
    exogs: list[str],
    eval_df: pl.DataFrame | None = None,
) -> pl.DataFrame:
    X_df = df.select(exogs).to_pandas()
    Y_df = df.select(endog).to_pandas()
    reg = LinearRegression(fit_intercept=True, n_jobs=-1)
    reg.fit(X_df, Y_df)
    if eval_df is None:
        eval_df = df
    else:
        missing_columns = set(exogs).difference(eval_df.columns)
        if len(missing_columns) > 0:
            raise ValueError(f"Missing columns in eval_df: {missing_columns}")

    Yhat = reg.predict(eval_df.select(exogs).to_pandas())
    schema = [endog] if isinstance(endog, str) else endog
    return pl.DataFrame(Yhat, schema=schema)
