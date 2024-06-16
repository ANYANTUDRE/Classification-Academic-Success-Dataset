import itertools

def feature_engineering(df, cat_cols):
    """
    This function is used for feature engineering
    :param df: the pandas dataframe with train/test data
    :param cat_cols: list of categorical columns
    :return: dataframe with new features
    """
    # this will create all 2-combinations of values in this list. for example:
    # list(itertools.combinations([1,2,3], 2)) will return [(1, 2), (1, 3), (2, 3)]

    combi = list(itertools.combinations(cat_cols, 2))
    for c1, c2 in combi:
        df.loc[:, c1 +"_"+ c2] = df[c1].astype(str) + "_" + df[c2].astype(str)
        
    return df