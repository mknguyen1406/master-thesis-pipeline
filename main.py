from classes.data_cleaning import DataCleaner
from classes.data_aggregating import DataAggregator
from classes.lda_modeling import LdaModel
from classes.dtm_modeling import DtmModel

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
        "bigram_min_count": 10,
        "dict_no_below": 10,
        "dict_no_above": 0.2
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

    aggregator_params = {
        "directory": "output/",
        "files": files,
        "file_suffix": "_clean.csv",
        "file_format": "csv",
        "date_col": "startDate",
        "target_cols": ["id", "startDate", "data_clean"]
    }

    aggregator = DataAggregator(**aggregator_params)
    # df_agg = aggregator.aggregate_data()
    # aggregator.save_to_csv(df_agg)

    #############################################################
    # Create doc-term-matrix
    #############################################################

    cleaner = DataCleaner(**cleaner_params)

    # Read already aggregated data
    df_agg = aggregator.read_final_from_csv()
    # df_agg = df_agg.sample(100).reset_index(drop=True)

    # Get converted cleaned data
    data_clean = cleaner.convert_data_clean(df_agg, "data_clean")

    # Create dictionary
    # dictionary = cleaner.create_dictionary(data_clean)
    # cleaner.save_dictionary(dictionary, "assets/dictionary_all_projects")

    # Load dictionary
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
        "time_ref_col": "fp",  # fp or year
        "path_to_dtm_binary": "assets/dtm-win64.exe",
        "dictionary": dictionary,
        "doc_term_matrix": doc_term_matrix,
        "seed": 0,
        "num_topics": 20,
        "output_file_path": "models/dtm/dtm_all_projects_20",
        "files": files
    }

    # Initialize model
    dtm_model = DtmModel(**model_params)

    # # Prepare data
    # df_year = dtm_model.prepare_data(df_agg)
    #
    # # print(df_year)
    #
    # # Train model
    # dtm_model.train_model()
    #
    # # Save model
    # dtm_model.save_model()

    # Print topics
    # dtm_top_terms = dtm_model.top_term_table(0, [1990, 2000, 2005, 2010, 2013], 10)

    # print(dtm_top_terms)

    #############################################################
    # Assign topics to documents
    #############################################################

    # Load previously saved model
    dtm_model.load_model()

    # # Get topic assignments per document
    # doc_topic_no = dtm_model.get_doc_topics(doc_term_matrix)
    #
    # # Get aggregated data
    # aggregator_params = {
    #     "directory": "data/",
    #     "files": files,
    #     "file_suffix": ".xlsx",
    #     "file_format": "xlsx",
    #     "sheet_name": "project",
    #     "date_col": "startDate",
    #     "target_cols": ["id", "title", "ecMaxContribution", "totalCost", "coordinatorCountry"],
    #     "dropna_cols": ["id", "startDate", "objective"]
    # }
    #
    # # Aggregate raw data with desired information
    # aggregator = DataAggregator(**aggregator_params)
    # df_info = aggregator.aggregate_data()
    #
    # # Add information to df and save data
    # df_output = df_agg.merge(df_info, how="left", on="id")
    # df_output["topic_no"] = doc_topic_no
    # df_output.to_csv("output/all_projects_topics.csv", index=False)

    # Generate excel file with topic data frames per
    # df_topics = dtm_model.generate_topic_tables()
    # df_topics.to_csv("output/topics/all_topics.csv", index=False, float_format="%.15f", sep=";", decimal=",")

    # Generate topic detail table
    dtm_model.generate_topic_detail_tables()

    # Calculate differences between topic time slices
    dtm_model.calculate_word_dif("output/topics/")

    #############################################################
    # Aggregate topic detail csv
    #############################################################

    topic_dif_files = [f"topic_dif_{i}" for i in range(20)]

    # Get aggregated data
    aggregator_params = {
        "directory": "output/topics/",
        "files": topic_dif_files,
        "file_suffix": ".csv",
        "file_format": "csv",
    }

    # Aggregate raw data with desired information
    aggregator = DataAggregator(**aggregator_params)
    df_topics_detail = aggregator.aggregate_data()
    df_topics_detail.to_csv("output/topics/all_topics_detail.csv", index=False, float_format="%.15f", sep=";",
                            decimal=",")

    ###############################################################################

    finish_time = datetime.datetime.now()

    print(f"##################################\nFinished main method. Total duration was {finish_time - start_time}")
