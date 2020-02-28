from classes.data_cleaning import DataCleaner
from classes.data_aggregating import DataAggregator
from classes.lda_modeling import LdaModel
from classes.dtm_modeling import DtmModel
from classes.dtm_evaluation import Evaluator
from classes.data_exploration import Explorer

from gensim import corpora

import datetime

if __name__ == '__main__':

    start_time = datetime.datetime.now()
    print(f"##################################\nStarting main method at {start_time}\n\n")

    #############################################################
    # Clean data
    #############################################################

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

    additional_stop_words = [
            "project",
            "new",
            "research",
            "high",
            "objective",
            "study",
            "approach",
            "result",
            "model",
            "different",
            "field",
            "use",
            "method",
            "analysis",
            "large",
            "potential",
            "order",
            "work",
            "main",
            "activity",
            "impact",
            "novel",
            "important",
            "key",
            "aim",
            "year",
            "proposal",
            "current",
            "major",
            "low",
            "goal",
            "problem",
            "effect",
            "term",
            "role",
            "solution",
            "scientific",
            "training",
            "researcher"
        ]

    cleaner_params = {
        "input_path": "data/",
        "output_path": "output/",
        "file_name": "<<PLACEHOLER>>",
        "sheet_name": "project",
        "date_col_index": ["startDate", "endDate"],
        "id_col": "id",
        "text_col": "objective",
        "date_col": "startDate",
        "keep_pos": ["PROPN", "NOUN", "ADJ"],
        "translate": False,
        "add_bigrams": True,
        "bigram_min_count": 50,
        "dict_no_below": 50,
        "dict_no_above": 0.999,
        "additional_stop_words": additional_stop_words
    }

    # for file_name in files:
    #
    #     print(f"###########################\nStart processing file {file_name} at {datetime.datetime.now()}")
    #
    #     cleaner_params["file_name"] = file_name
    #
    #     # Initialize cleaner
    #     cleaner = DataCleaner(**cleaner_params)
    #
    #     # Read data
    #     df_raw = cleaner.read_data()
    #
    #     # Only take small sample
    #     # df_raw = df_raw[:10]
    #
    #     try:
    #         # Clean data
    #         df_clean, data_clean = cleaner.clean_data(df_raw)
    #
    #         # Add bigrams
    #         data_clean = cleaner.add_bigrams_to_data(data_clean)
    #
    #         # Add clean data to data frame
    #         df_clean["data_clean"] = data_clean
    #
    #         # Save cleaned data frame
    #         cleaner.save_df_clean(df_clean)
    #
    #     except:
    #         print(f"#######################################\nError at file {file_name}")
    #
    #     print(f"###########################\nDone processing file {file_name} at {datetime.datetime.now()}\n\n")

    #############################################################
    # Aggregate data
    #############################################################

    # finish_time = datetime.datetime.now()
    # print(f"##################################\nFinished data cleaning. Total duration was {finish_time - start_time}\n")

    aggregator_params = {
        "directory": "output/",
        "files": files,
        "file_suffix": "_clean.csv",
        "file_format": "csv",
        "date_col": "startDate",
        "target_cols": ["id", "startDate", "data_clean", "fp", "fp_no"]
    }

    aggregator = DataAggregator(**aggregator_params)

    # df_agg = aggregator.aggregate_data()
    # aggregator.save_to_csv(df_agg)

    #############################################################
    # Create doc-term-matrix
    #############################################################

    finish_time = datetime.datetime.now()
    print(
        f"##################################\nFinished data aggregation. Total duration was {finish_time - start_time}\n")

    # Read already aggregated data
    df_agg = aggregator.read_final_from_csv()
    # df_agg = df_agg.sample(100).reset_index(drop=True)

    # Get converted cleaned data
    cleaner = DataCleaner(**cleaner_params)
    data_clean = cleaner.convert_data_clean(df_agg, "data_clean")

    #############################################################
    # Explore corpus
    #############################################################

    explorer = Explorer(data_clean)
    explorer.calculate_word_df()

    df_get_n_most_frequent_words = explorer.get_n_most_frequent_words(10)
    df_get_most_frequent_words_by_frac = explorer.get_most_frequent_words_by_frac(0.2)
    df_get_least_frequent_words_by_doc_count = explorer.get_least_frequent_words_by_doc_count(10)

    print("df_get_n_most_frequent_words:\n", df_get_n_most_frequent_words.head(100), "\nshape:\n", df_get_n_most_frequent_words.shape, "\n")
    print("df_get_most_frequent_words_by_frac:\n", df_get_most_frequent_words_by_frac.head(100), "shape:\n", df_get_most_frequent_words_by_frac.shape, "\n")
    print("df_get_least_frequent_words_by_doc_count:\n", df_get_least_frequent_words_by_doc_count.head(100), "shape:\n", df_get_least_frequent_words_by_doc_count.shape, "\n")

    print(explorer.df_words[explorer.df_words["count"] == 1].shape)
    print(explorer.df_words[explorer.df_words["doc_count"] == 1].shape)

    print(explorer.df_words.groupby(["count"]).count().shape)
    print(explorer.df_words.groupby(["doc_count"]).count().shape)

    df_get_n_most_frequent_words = explorer.get_n_most_frequent_words(100)
    # df_get_n_most_frequent_words.to_excel("output/top_words.xlsx", index=False)

    explorer.df_words.to_excel("all_words_table.xlsx")

    ###############################################################################

    finish_time = datetime.datetime.now()
    print(f"##################################\nFinished main method. Total duration was {finish_time - start_time}")
