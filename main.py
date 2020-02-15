from classes.data_cleaning import DataCleaner
from classes.lda_modeling import LdaModel
from classes.dtm_modeling import DtmModel

import datetime

if __name__ == '__main__':

    #############################################################
    # Clean data
    #############################################################

    # Define all files to be processed
    files = ["h2020_projects",
             "fp7_projects",
             "fp6_projects",
             "fp5_projects",
             "fp4_projects",
             "fp3_projects",
             "fp2_projects",
             "fp1_projects",
             ]

    for file_name in files:

        print(f"###########################\nStart processing file {file_name} at {datetime.datetime.now()}")

        cleaner_params = {
            "input_path": "data/",
            "output_path": "output/",
            "file_name": file_name,
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

        # Initialize cleaner
        cleaner = DataCleaner(**cleaner_params)

        # Read data
        df_raw = cleaner.read_data()

        # Only take small sample
        # df_raw = df_raw[:10]

        try:
            # Clean data
            df_clean, data_clean = cleaner.clean_data(df_raw)

            # Add bigrams
            data_clean = cleaner.add_bigrams_to_data(data_clean)

            # Add clean data to data frame
            df_clean["data_clean"] = data_clean

            # Save cleaned data frame
            cleaner.save_df_clean(df_clean)

        except:
            print(f"#######################################\nError at file {file_name}")

        print(f"###########################\nDone processing file {file_name} at {datetime.datetime.now()}\n\n")

        # df_clean = pd.DataFrame({"id": df_raw[cleaner_params["id_col"]],
        #                          "startDate": df_raw[cleaner_params["date_col"]],
        #                          "objective": df_raw[cleaner_params["text_col"]],
        #                          "data_clean": data_clean})

        # Create dictionary and doc-term-matrix
        # dictionary, dtm = cleaner.create_dictionary(data_clean)

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

        # model_params = {
        #     "date_col": "startDate",
        #     "path_to_dtm_binary": "assets/dtm-win64.exe",
        #     "dictionary": dictionary,
        #     "doc_term_matrix": dtm,
        #     "seed": 0,
        #     "num_topics": 5,
        #     "output_path": "output/dtm_model_h2020",
        # }
        #
        # # Initialize model
        # dtm_model = DtmModel(**model_params)
        #
        # # Prepare data
        # df_year = dtm_model.prepare_data(df_raw)
        #
        # print(df_year)

        # # Train model
        # dtm_model.train_model()

        # Print topics
        # dtm_top_terms = dtm_model.top_term_table(0, [2007, 2008, 2009, 2010, 2011, 2012, 2013], 10)

        # print(dtm_top_terms)
