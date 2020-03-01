from classes.data_cleaning import DataCleaner
from classes.data_aggregating import DataAggregator
from classes.lda_modeling import LdaModel
from classes.dtm_modeling import DtmModel
from classes.dtm_evaluation import Evaluator
from classes.data_exploration import Explorer


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
        "id_col": "rcn",
        "text_col": "objective",
        "date_col": "startDate",
        "keep_pos": ["PROPN", "NOUN", "ADJ"],
        "translate": False,
        "add_bigrams": True,
        "bigram_min_count": 50,
        "dict_no_below": 20,
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

    finish_time = datetime.datetime.now()
    print(f"##################################\nFinished data cleaning. Total duration was {finish_time - start_time}\n")

    aggregator_params = {
        "directory": "output/",
        "files": files,
        "file_suffix": "_clean.csv",
        "file_format": "csv",
        "date_col": "startDate",
        "target_cols": ["rcn", "startDate", "data_clean", "fp", "fp_no"]
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

    # Create dictionary
    dictionary = cleaner.create_dictionary(data_clean)
    # cleaner.save_dictionary(dictionary, "assets/dictionary_all_projects")

    # # Load dictionary
    dictionary = cleaner.load_dictionary("assets/dictionary_all_projects")

    # Create dtm
    doc_term_matrix = cleaner.create_doc_term_matrix(dictionary, data_clean)

    #############################################################
    # Build LDA model
    #############################################################

    # model_params = {
    #     "dictionary": dictionary,
    #     "doc_term_matrix": dtm,
    #     "seed": 0,
    #     "num_topics": 5
    # }
    #
    # # Initialize model
    # lda_model = LdaModel(**model_params)
    #
    # # Train model
    # lda_model.train_model()
    #
    # # Print topics
    # topics_df = lda_model.get_top_n_terms(5)
    #
    # print(topics_df)

    #############################################################
    # Build DTM model
    #############################################################

    # cleaner_params = {
    #     "input_path": "data/",
    #     "output_path": "output/",
    #     "file_name": "",
    #     "sheet_name": "project",
    #     "date_col_index": ["startDate", "endDate"],
    #     "id_col": "id",
    #     "text_col": "objective",
    #     "date_col": "startDate",
    #     "keep_pos": ["PROPN", "NOUN", "ADJ"],
    #     "translate": False,
    #     "add_bigrams": True,
    #     "bigram_min_count": 10,
    #     "dict_no_below": 10,
    #     "dict_no_above": 0.2
    # }

    model_params = {
        "date_col": "startDate",
        "time_ref_col": "fp_no",  # fp or year
        "path_to_dtm_binary": "assets/dtm-win64.exe",
        "dictionary": dictionary,
        "doc_term_matrix": doc_term_matrix,
        "seed": 0,
        "num_topics": 20,
        "output_file_path": "models/dtm/200228_dtm_all_projects_20",
        "files": files
    }

    # Initialize model
    dtm_model = DtmModel(**model_params)

    # # Prepare data
    # df_year = dtm_model.prepare_data(df_agg)
    #
    # # Train model
    # dtm_model.train_model()
    #
    # # Save model
    # dtm_model.save_model()

    #############################################################
    # Assign topics and programmes to documents
    #############################################################

    finish_time = datetime.datetime.now()
    print(
        f"##################################\nFinished modeling. Total duration was {finish_time - start_time}\n")

    # Load previously saved model
    dtm_model.load_model()

    # Get topic assignments per document and save
    df_agg_topics = dtm_model.get_doc_topics(doc_term_matrix, df_agg)

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

    # # Make sure data type of join key is identical and join info
    # df_info["rcn"] = df_info["rcn"].apply(str)
    # df_agg_topics["rcn"] = df_agg_topics["rcn"].apply(str)
    # df_join = df_agg_topics.merge(df_info, how="left", on="rcn")
    #
    # # Save result
    # df_join.to_excel("output/all_projects_topics.xlsx", index=False, float_format="%.15f")
    #
    # # Generate stacked project topic table
    # df_project_topics = dtm_model.generate_project_topic_table(df_join)
    # df_project_topics.to_csv("output/project_topics_stacked.csv", float_format="%.15f", sep=";", decimal=",")

    # # Generate excel file with topic data frames per
    # df_topics = dtm_model.generate_topic_tables()
    # df_topics.to_csv("output/topics/all_topics.csv", index=False, float_format="%.15f", sep=";", decimal=",")
    #
    # # Generate topic detail table
    # dtm_model.generate_topic_detail_tables()
    #
    # # Calculate differences between topic time slices
    # dtm_model.calculate_word_dif("output/topics/")
    #
    # #############################################################
    # # Aggregate topic detail csv
    # #############################################################
    #
    # topic_dif_files = [f"topic_dif_{i}" for i in range(20)]
    #
    # # Get aggregated data
    # aggregator_params = {
    #     "directory": "output/topics/",
    #     "files": topic_dif_files,
    #     "file_suffix": ".csv",
    #     "file_format": "csv",
    # }
    #
    # # Aggregate raw data with desired information
    # aggregator = DataAggregator(**aggregator_params)
    # df_topics_detail = aggregator.aggregate_data()
    # df_topics_detail.to_csv("output/topics/all_topics_detail.csv", index=False, float_format="%.15f", sep=";",
    #                         decimal=",")

    #############################################################
    # Calculate evaluation metrics
    #############################################################

    finish_time = datetime.datetime.now()
    print(
        f"##################################\nFinished calculating Power BI outputs. Total duration was {finish_time - start_time}\n")

    # evaluator_params = {
    #     "topic_dif_file_path": "output/topics/all_topics_detail.csv",
    #     "files": files
    # }
    #
    # evaluator = Evaluator(**evaluator_params)
    #
    # # For time slice lag 1
    # evaluator.transform_topic_dif_data_lag_1()
    #
    # # Accuracy
    # tp_tn = evaluator.df_eval["binary_lag_1_tp_tn"].sum()
    # total = evaluator.df_eval.shape[0]
    # print(f"Accuracy of binary predictions with time lag 1: \n{tp_tn/total}")
    #
    # # Correlation
    # corr = evaluator.df_eval.corr().loc["real_lag_1_pred", "real_lag_1_test"]
    # print(f"Correlation coefficient between between topic dif values with time lag 1: \n{corr}\n")
    #
    # # For time slice lag 2
    # evaluator.transform_topic_dif_data_lag_2()
    #
    # # Accuracy
    # tp_tn = evaluator.df_eval["binary_lag_2_tp_tn"].sum()
    # total = evaluator.df_eval.shape[0]
    # print(f"Accuracy of binary predictions with time lag 2: \n{tp_tn / total}")
    #
    # # Correlation
    # corr = evaluator.df_eval.corr().loc["real_lag_2_pred", "real_lag_2_test"]
    # print(f"Correlation coefficient between between topic dif values with time lag 2: \n{corr}")

    #############################################################
    # Explore corpus
    #############################################################

    # explorer = Explorer(data_clean)
    # explorer.calculate_word_df()
    #
    # df_get_n_most_frequent_words = explorer.get_n_most_frequent_words(10)
    # df_get_most_frequent_words_by_frac = explorer.get_most_frequent_words_by_frac(0.2)
    # df_get_least_frequent_words_by_doc_count = explorer.get_least_frequent_words_by_doc_count(10)
    #
    # print("df_get_n_most_frequent_words:\n", df_get_n_most_frequent_words.head(100), "\nshape:\n", df_get_n_most_frequent_words.shape, "\n")
    # print("df_get_most_frequent_words_by_frac:\n", df_get_most_frequent_words_by_frac.head(100), "shape:\n", df_get_most_frequent_words_by_frac.shape, "\n")
    # print("df_get_least_frequent_words_by_doc_count:\n", df_get_least_frequent_words_by_doc_count.head(100), "shape:\n", df_get_least_frequent_words_by_doc_count.shape, "\n")
    #
    # print(explorer.df_words[explorer.df_words["count"] == 1].shape)
    # print(explorer.df_words[explorer.df_words["doc_count"] == 1].shape)
    #
    # print(explorer.df_words.groupby(["count"]).count().shape)
    # print(explorer.df_words.groupby(["doc_count"]).count().shape)
    #
    # df_get_n_most_frequent_words = explorer.get_n_most_frequent_words(100)
    # df_get_n_most_frequent_words.to_excel("output/top_words.xlsx", index=False)

    ###############################################################################

    finish_time = datetime.datetime.now()
    print(f"##################################\nFinished main method. Total duration was {finish_time - start_time}")
