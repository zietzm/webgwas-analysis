import re
from pathlib import Path
from typing import IO

import polars as pl
from pydantic import BaseModel, Field

PathType = str | Path | IO[str] | IO[bytes] | bytes


class PhecodeDefinition(BaseModel):
    phecode: str
    include_icd: list[str] = Field(default_factory=list)
    exclude_icd: list[str] = Field(default_factory=list)


def load_inclusions(map_file: PathType) -> dict[str, list[str]]:
    """Load a mapping from phecode to all inclusion ICD codes.

    Example output:
        {
            '297': ['E989', 'E988', 'E985', 'E986'],
            '967': ['E853', 'E937', 'E851', 'E852', 'E938'],
            ...
        }
    """

    return (
        pl.read_csv(map_file, schema_overrides={"Phecode": pl.String, "ICD": pl.String})
        .rename({"Phecode": "phecode"})
        .select(pl.col("phecode", "ICD").str.strip_chars(" "))
        .group_by("phecode")
        .agg(pl.col("ICD").implode().flatten())
        .with_columns(pl.col("ICD").list.unique())
        .to_pandas()
        .set_index("phecode")["ICD"]
        .to_dict()
    )


def load_exclusion_ranges(definition_file: PathType) -> dict[str, list[str]]:
    """Load a mapping from phecode to all exclusion ranges.

    Example output:
        {
            '136': ['130-136.99'],
            '145': ['140-149.99', '210-210.99'],
            '1019': [''],
            ...
        }
    """
    return (
        pl.read_csv(
            definition_file,
            schema_overrides={"phecode": pl.String},
        )
        .select(
            pl.col("phecode").str.strip_chars(" "),
            pl.col("phecode_exclude_range")
            .str.split(",")
            .list.eval(pl.element().str.strip_chars(" "))
            .list.set_difference(pl.lit([""])),
        )
        .to_pandas()
        .set_index("phecode")["phecode_exclude_range"]
        .to_dict()
    )


def phecode_range_to_phecodes(
    phecode_range: str,
    unique_phecodes: list[str],
) -> list[str]:
    """Parse a phecode range and find all matching phecodes

    Examples:
        phecode_range_to_phecodes(
            phecode_range="1-2",
            unique_phecodes=["1", "2", "3", "4"],
        )
        # Result: ["1", "2"]
    """

    match = re.match(r"^([\d\.]+)-([\d\.]+)$", phecode_range)
    if match is None:
        raise ValueError(f"Invalid phecode range: `{phecode_range}`")

    start, end = match.groups()
    start_float = float(start)
    end_float = float(end)
    if start_float > end_float:
        raise ValueError(f"Invalid phecode range: `{phecode_range}`")

    return [
        phecode
        for phecode in unique_phecodes
        if start_float <= float(phecode) <= end_float
    ]


def load_definitions(
    definition_file: PathType,
    map_file: PathType,
) -> list[PhecodeDefinition]:
    phecode_to_include_icd = load_inclusions(map_file)
    phecode_to_exclude_icd = load_exclusion_ranges(definition_file)

    unique_phecodes = list(phecode_to_include_icd.keys())

    results = list()
    for phecode, inclusions in phecode_to_include_icd.items():
        # Find all phecodes that are excluded by this phecode
        exclusion_ranges = phecode_to_exclude_icd.get(phecode, [])
        excluded_phecodes = set()
        for exclusion_range in exclusion_ranges:
            these_excluded_phecodes = phecode_range_to_phecodes(
                exclusion_range, unique_phecodes
            )
            excluded_phecodes.update(these_excluded_phecodes)

        # Make sure the phecode itself is not excluded
        excluded_phecodes.discard(phecode)

        # Take the union of all inclusion codes for all excluded phecodes.
        # These are all ICD codes that lead to exclusion here.
        excluded_icd = set()
        for excluded_phecode in excluded_phecodes:
            these_excluded_icd = phecode_to_include_icd.get(excluded_phecode, [])
            excluded_icd.update(these_excluded_icd)

        result = PhecodeDefinition(
            phecode=phecode,
            include_icd=inclusions,
            exclude_icd=list(excluded_icd),
        )
        results.append(result)

    return results


def filter_definitions(
    definitions: list[PhecodeDefinition],
    icd_codes: set[str],
) -> list[PhecodeDefinition]:
    """Update phecode definitions to only include ICD codes that are in the
    given set. Remove trivial definitions that are just a single ICD code.
    """

    results = list()
    for definition in definitions:
        updated_inclusion = [
            code for code in definition.include_icd if code in icd_codes
        ]
        updated_exclusion = [
            code for code in definition.exclude_icd if code in icd_codes
        ]

        # Conditions for excluding a definition
        no_inclusion = len(updated_inclusion) == 0
        trivial = len(updated_inclusion) == 1 and len(updated_exclusion) == 0

        if no_inclusion or trivial:
            continue

        result = PhecodeDefinition(
            phecode=definition.phecode,
            include_icd=updated_inclusion,
            exclude_icd=updated_exclusion,
        )
        results.append(result)

    return results


def check_definitions(
    definitions: list[PhecodeDefinition], available_icd_codes: set[str]
):
    """Check that all phecode definitions are valid."""
    definition_codes = set()
    for definition in definitions:
        definition_codes.update(definition.include_icd)
        definition_codes.update(definition.exclude_icd)

    missing_codes = definition_codes - available_icd_codes
    if len(missing_codes) > 0:
        raise ValueError(f"Missing ICD codes: {missing_codes}")


def check_phenotype_ranges(
    phenotype_df: pl.DataFrame, min_: float = 0, max_: float = 1
):
    """Check that all phenotypes in the dataframe are in the given range."""

    error_phenotypes = (
        phenotype_df.unpivot()
        .group_by("variable")
        .agg(
            pl.col("value").min().alias("min"),
            pl.col("value").max().alias("max"),
        )
        .filter(pl.col("min").lt(min_) | pl.col("max").gt(max_))["variable"]
        .to_list()
    )
    if len(error_phenotypes) > 0:
        raise ValueError(
            f"Phenotypes must be in range [{min_}, {max_}]: {error_phenotypes}"
        )


def check_phenotypes_are_binary(phenotype_df: pl.DataFrame):
    """Check that all phenotypes in the dataframe are binary."""

    error_phenotypes = (
        phenotype_df.unpivot()
        .group_by("variable")
        .agg(
            pl.col("value").min().alias("min"),
            pl.col("value").max().alias("max"),
            pl.col("value").n_unique().alias("n_unique"),
        )
        .filter(pl.col("min").ne(0) | pl.col("max").ne(1) | pl.col("n_unique") > 2)
        .select("variable")["variable"]
        .to_list()
    )
    if len(error_phenotypes) > 0:
        raise ValueError(f"Phenotypes must be binary: {error_phenotypes}")


def apply_definitions(
    definitions: list[PhecodeDefinition],
    icd_df: pl.DataFrame,
) -> pl.DataFrame:
    """Apply phecode definitions to an ICD code dataframe.

    The dataframe should have only columns corresponding to ICD-9/10 codes, named
    accordingly.
    """

    check_definitions(definitions, set(icd_df.columns))
    check_phenotypes_are_binary(icd_df)

    col_definitions = list()
    for definition in definitions:
        included = pl.max_horizontal(definition.include_icd).eq(1)
        if len(definition.exclude_icd) == 0:
            col = included
        else:
            excluded = pl.max_horizontal(definition.exclude_icd).eq(1)
            col = pl.when(included & excluded).then(None).otherwise(included)

        col = col.cast(pl.Int8).alias(definition.phecode)
        col_definitions.append(col)

    return icd_df.select(col_definitions)


def apply_definitions_fuzzy(
    definitions: list[PhecodeDefinition],
    icd_df: pl.DataFrame,
) -> pl.DataFrame:
    """Apply phecode definitions to an ICD code dataframe using fuzzy logic.

    This is fundamentally different from `apply_definitions` in that it
    never assigns a missing value to a person. Instead, it approximates each
    phecode as `AND(included, NOT(excluded))`. The resulting definitions are
    floating point numbers.
    """

    check_definitions(definitions, set(icd_df.columns))
    check_phenotype_ranges(icd_df)

    col_definitions = list()
    for definition in definitions:
        included = pl.max_horizontal(definition.include_icd)
        if len(definition.exclude_icd) == 0:
            col = included
        else:
            excluded = pl.max_horizontal(definition.exclude_icd)
            not_excluded = pl.lit(1).sub(excluded)
            col = pl.min_horizontal([included, not_excluded])

        col = col.alias(definition.phecode)
        col_definitions.append(col)

    return icd_df.select(col_definitions)


def filter_phenotypes(
    phenotype_df: pl.DataFrame, min_n_cases: int = 10
) -> pl.DataFrame:
    """Drop all phecodes with fewer than `min_n_cases` cases.

    Dataframe should have only columns corresponding to phecodes, with column
    names corresponding to phecodes (e.g. `008.5`).
    """

    phecodes_to_keep = (
        phenotype_df.select(pl.all().sum())
        .unpivot()
        .filter(pl.col("value") >= min_n_cases)["variable"]
        .to_list()
    )
    return phenotype_df.select(phecodes_to_keep)
