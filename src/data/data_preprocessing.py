import os
import yaml

import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def construct_preprocessor(preprocessor_cfg):
    preprocessing_pipeline = []

    for column, settings in preprocessor_cfg.items():   
        selected_imputer = settings["imputer"]
        selected_transformer = settings["transformer"]

        column_steps = []

        if selected_imputer in ["mean", "median", "most_frequent"]:
            column_steps.append(("imputer", SimpleImputer(strategy=selected_imputer)))

        if selected_transformer == "standard":
            column_steps.append(("transformer", StandardScaler()))
        elif selected_transformer == "MinMax":
            column_steps.append(("transformer", MinMaxScaler()))
        elif selected_transformer == "oh_encoder":
            column_steps.append(("transformer", OneHotEncoder(sparse_output=False)))
        elif selected_transformer == "ordinal":
            column_steps.append(("transformer", OrdinalEncoder()))

        preprocessing_pipeline.append((column, Pipeline(column_steps), [column]))

    return ColumnTransformer(preprocessing_pipeline, verbose_feature_names_out=False).set_output(transform="pandas")


def preprocess_data(df, cfg, preprocessor=None):
    """
    Preprocess the data according to the configuration file
    
    :param df: pd.DataFrame - dataframe to preprocess

    :param cfg: dict - transformation instructions

    :param preprocessor: ColumnTransformer - preprocessor to use. If None, a new preprocessor will be created

    :return:
        df_preprocessed: pd.DataFrame - preprocessed dataframe

        preprocessor: ColumnTransformer - preprocessor used to preprocess the data

    :raises:
        AssertionError: if the columns in the dataframe do not match the columns in the config file
    """

    # Ensure df has all expected columns (no missing or extra columns)
    assert set(df.columns) == set(cfg.keys()), "Columns in the dataframe do not match the columns in the config file"

    if preprocessor is None:
        preprocessor = construct_preprocessor(cfg)
        preprocessed_df = preprocessor.fit_transform(df)
    else:
        preprocessed_df = preprocessor.transform(df)

    return preprocessed_df, preprocessor







