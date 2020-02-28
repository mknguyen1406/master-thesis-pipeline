from classes.data_cleaning import DataCleaner
from classes.data_aggregating import DataAggregator
from classes.lda_modeling import LdaModel
from classes.dtm_modeling import DtmModel
from classes.dtm_evaluation import Evaluator
# from classes.data_exploration import Explorer

from gensim import corpora

import datetime

if __name__ == '__main__':
    # Define all files to be processed
    files = ["fp1_projects",
             "fp2_projects",
             "fp3_projects",
             "fp4_projects",
             "fp5_projects",
             "fp6_projects",
             "fp7_projects",
             "h2020_projects"
             ]

    # Get aggregated data
    aggregator_params = {
        "directory": "data/",
        "files": files,
        "file_suffix": ".xlsx",
        "file_format": "xlsx",
        "sheet_name": "project",
        "date_col": "startDate",
        "target_cols": ["rcn", "title", "ecMaxContribution", "totalCost", "coordinatorCountry"],
        "dropna_cols": ["rcn", "startDate", "objective"]
    }

    # Aggregate raw data with desired information
    aggregator = DataAggregator(**aggregator_params)
    df_info = aggregator.aggregate_data()

    print(df_info.head())
