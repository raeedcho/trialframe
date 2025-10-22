import pandas as pd

def get_index_level(df,level=None):
    if level is None:
        level = df.index.names
    return df.reset_index(level=level)[level]

def multivalue_xs(df: pd.DataFrame,keys: list,level,axis:int=0,**kwargs) -> pd.DataFrame:
    if axis == 1:
        df = df.T

    possible_keys = df.groupby(level=level).groups.keys()
    ret_df = pd.concat([df.xs(key=key,level=level,drop_level=False,**kwargs) for key in keys if key in possible_keys])
    
    if axis == 1:
        ret_df = ret_df.T

    return ret_df

def hierarchical_assign(df,assign_dict):
    '''
    Extends pandas.DataFrame.assign to work with hierarchical columns

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to assign to
    assign_dict : dict of pandas.DataFrame or callable
        dictionary of dataframes to assign to df
    '''
    # Keep original columns as-is (flat or multiindex). We'll append hierarchical blocks.

    # Build a hierarchical (signal, channel) column block to append
    frames = []
    for key, val in assign_dict.items():
        res: pd.DataFrame = val(df) if callable(val) else val  # type: ignore[assignment]
        # Ensure index alignment
        res = res.reindex(df.index)
        # Reduce to single column level so we can add ('signal','channel') on top
        if isinstance(res.columns, pd.MultiIndex):
            res = res.copy()
            res.columns = res.columns.get_level_values(-1)
            res.columns.name = 'channel'
        frames.append(res)

    right = pd.concat(
        frames,
        axis=1,
        keys=list(assign_dict.keys()),
        names=['signal', 'channel'],
    )
    # Concatenate along columns to avoid merge-level mismatch issues
    return pd.concat([df, right], axis=1)