import pandas as pd

def extract_data(cfg):
    """
    Extract the data according to the configuration file
    
    :param cfg: dict - data_management configuration file
    
    :return:
        X: pd.DataFrame - raw dataset
    """
    df = pd.read_csv(cfg["data_source"])

    return df