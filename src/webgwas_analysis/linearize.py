import polars as pl
import tqdm.auto as tqdm
from sklearn.linear_model import LinearRegression


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
        if not eval_df.columns == df.columns:
            raise ValueError("df and eval_df must have the same columns")

    Yhat = reg.predict(eval_df.select(exogs).to_pandas())
    schema = [endog] if isinstance(endog, str) else endog
    return pl.DataFrame(Yhat, schema=schema)


def approximate_all(
    df: pl.DataFrame,
    endogs: list[str],
    exogs: list[str],
    eval_df: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Approximate all phenotypes in the dataframe using linear regression."""

    if eval_df is not None and not df.columns == eval_df.columns:
        raise ValueError("df and eval_df must have the same columns")

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

    if eval_df is not None and not df.columns == eval_df.columns:
        raise ValueError("df and eval_df must have the same columns")

    approx_df = dict()
    for endog in tqdm.tqdm(endogs):
        this_df = df.select([endog] + exogs).drop_nulls()
        Yhat_df = _approximate(this_df, endog, exogs, eval_df)
        assert Yhat_df.shape[1] == 1
        approx_df[endog] = Yhat_df[endog]

    return pl.DataFrame(approx_df)
