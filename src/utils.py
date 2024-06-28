import itertools
import pandas as pd
import config


def feature_engineering(df):
    df['age_group'] = pd.cut(df['Age at enrollment'], bins=[17, 20, 25, 30, 35, 40, 50], labels=[1, 2, 3, 4, 5, 6])
    # Creating interaction features
    df['age_admission_interaction'] = df['Age at enrollment'] * df['Admission grade'] # to keep
    df['parental_education_interaction'] = df['Mother\'s qualification'] * df['Father\'s qualification'] # to keep
    df['parental_occupation_interaction'] = df['Mother\'s occupation'] * df['Father\'s occupation']
    #df['displaced_curricular_interaction'] = df['Displaced'] * df['Curricular units 1st sem (approved)'] # improve only the 2 first folds

    df.drop('Age at enrollment', inplace=True, axis=1)
    return df


def feature_engineering_auto(df, cat_cols):
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


def test_mean_target_encoding(train, test, alpha=5):
    """
        function to target encode the test dataset
    """
    global_mean = train["Target"].mean()
    num_classes = train["Target"].nunique()

    # list of numerical columns
    cat_cols = ['Marital status',
                'Application mode',
                #'Course',
                #'Previous qualification', #no
                'Nacionality',   # no but can be interesting
                "Mother's qualification", 
                #"Father's qualification", 
                "Mother's occupation",
                ]

    # for all feature columns, i.e. categorical columns
    for column in cat_cols:
        # create dict of category:mean target
        category_sum = train.groupby(column)["Target"].sum()
        category_size = train.groupby(column).size()
        # Calculate smoothed mean target statistics
        train_statistics = (category_sum + global_mean * alpha) / (category_size + alpha)

        # Map the encoded values to test data
        for cls in range(num_classes):
            enc_col_name = f"{column}_enc_class_{cls}"
            # Use .loc to avoid SettingWithCopyWarning
            test.loc[:, enc_col_name] = test[column].map(lambda x: train_statistics.get(x, global_mean)).fillna(global_mean)

    return test


def mean_target_encoding(train, alpha=5):
    global_mean = train["Target"].mean()
    num_classes = train["Target"].nunique()

    # list of numerical columns
    cat_cols = ['Marital status',
                'Application mode',
                #'Course',
                #'Previous qualification', #no
                'Nacionality',   # no but can be interesting
                "Mother's qualification", 
                #"Father's qualification", 
                "Mother's occupation",
                ]

    # a list to store 5 validation dataframes
    encoded_dfs = []
    
    for fold in range(config.N_FOLDS):
        df_train = train[train.kfold != fold].reset_index(drop=True)
        df_valid = train[train.kfold == fold].reset_index(drop=True)

        for column in cat_cols:
            category_sum = df_train.groupby(column)["Target"].sum()
            category_size = df_train.groupby(column).size()
            # Calculate smoothed mean target statistics
            train_statistics = (category_sum + global_mean * alpha) / (category_size + alpha)
            #print(train_statistics)
            # Map the encoded values to validation data
            for cls in range(num_classes):
                enc_col_name = f"{column}_enc_class_{cls}"
                # Use .loc to avoid SettingWithCopyWarning
                df_valid.loc[:, enc_col_name] = df_valid[column].map(lambda x: train_statistics.get(x, global_mean)).fillna(global_mean)
        
        # append to our list of encoded validation dataframes
        encoded_dfs.append(df_valid)

    # create full data frame again and return
    encoded_df = pd.concat(encoded_dfs, axis=0)
    return encoded_df
