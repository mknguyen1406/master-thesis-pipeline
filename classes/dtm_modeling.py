import pandas as pd
import numpy as np

from gensim.models.wrappers import DtmModel as DtmModelClass

import pickle # to save data to files for later use
import datetime

from gensim.test.utils import common_corpus, common_dictionary


class DtmModel:

    def __init__(self, date_col, time_ref_col, path_to_dtm_binary, dictionary, doc_term_matrix, seed, num_topics,
                 output_file_path, files):

        self.date_col = date_col
        self.time_ref_col = time_ref_col
        self.path_to_dtm_binary = path_to_dtm_binary
        self.dictionary = dictionary
        self.doc_term_matrix = doc_term_matrix
        self.seed = seed
        self.num_topics = num_topics
        self.output_file_path = output_file_path
        self.files = files

        self.time_slice_labels = None
        self.time_slices = None
        self.model = None
        self.topic_df_list = None

    def prepare_data(self, df):

        # Add year column to data frame
        def get_year(x):
            return x.year

        yrs = df[self.date_col].apply(lambda x: get_year(x))
        df["year"] = yrs

        # Get time slice labels
        self.time_slice_labels = df[self.time_ref_col].unique()
        self.time_slices = df.groupby(self.time_ref_col).size()

        print("Time_slices\n", self.time_slices)

        return df

    def train_model(self):

        # train DTM model
        print("Start time of DTM training: {}".format(datetime.datetime.now()))

        self.model = DtmModelClass(
            self.path_to_dtm_binary, corpus=self.doc_term_matrix, id2word=self.dictionary,
            time_slices=self.time_slices,
            num_topics=self.num_topics,
            rng_seed=self.seed
        )

        print("End time of DTM training: {}".format(datetime.datetime.now()))

    def save_model(self):

        # Save to file
        self.model.save(self.output_file_path)

        print(f"Dynamic topic model saved to {self.output_file_path}")

    def load_model(self):

        # Load model
        self.model = DtmModelClass.load(self.output_file_path)

        print(f"Model loaded from {self.output_file_path}")

    def top_term_table(self, topic, slices, topn=10):
        """Returns a dataframe with the top n terms in the topic for each of
        the given time slices."""

        data = {"Topic_ID": [topic] * topn, "Word_Rank": [i for i in range(topn)]}
        for time_slice in slices:
            time = np.where(self.time_slice_labels == time_slice)[0][0]
            data[time_slice] = [
                term for p, term
                in self.model.show_topic(topic, time=time, topn=topn)
            ]
        df = pd.DataFrame(data)
        return df

    def get_doc_topics(self, doc_term_matrix, df_agg):

        # Get topic assignment for each document
        doc_topic, topic_term, doc_lengths, term_frequency, vocab = self.model.dtm_vis(doc_term_matrix, 0)

        # Create topic label vector
        doc_topic_no = [np.argmax(array) for array in doc_topic]

        # Create document topic matrix
        topic_cols = [
            "topic_0",
            "topic_1",
            "topic_2",
            "topic_3",
            "topic_4",
            "topic_5",
            "topic_6",
            "topic_7",
            "topic_8",
            "topic_9",
            "topic_10",
            "topic_11",
            "topic_12",
            "topic_13",
            "topic_14",
            "topic_15",
            "topic_16",
            "topic_17",
            "topic_18",
            "topic_19"
        ]

        df_doc_topic = pd.DataFrame(doc_topic, columns=topic_cols)
        df_doc_topic["topic_no"] = doc_topic_no

        df_output = pd.concat([df_agg, df_doc_topic], axis=1)

        return df_output

    def generate_topic_tables(self):
        """
        Generate a list with a data frame for each topic, where rows denote a word and columns a time slice.
        :param files: Needed for the column names of the data frames
        :return: List of data frames for each topic
        """

        time_slices = self.files

        # topic_df_list = []

        # Gather data for each words in each topic in each time slice
        all_topics = []

        # For each time slice
        for time_id in range(len(time_slices)):

            def safe_div(x, y):
                if y == 0:
                    return 0
                return x / y

            time = time_slices[time_id]

            # Create data frame with dummy column having the length of the vocab
            # df_topic = pd.DataFrame([0] * len(vocab))

            # Get all topic-word distributions for time slice i
            _, topic_term, _, _, vocab = self.model.dtm_vis(self.doc_term_matrix, time_id)

            for topic_id in range(len(topic_term)):

                # Topic-word distribution for one topic at time slice i
                topic_at_time_slice = topic_term[topic_id]

                # For each word in this topic
                for word_id in range(len(topic_at_time_slice)):

                    # Gather all data records
                    data_word = vocab[word_id]
                    data_topic = topic_id
                    data_time = time
                    data_time_no = time_id
                    data_load = topic_at_time_slice[word_id]

                    # Calculate difference of word load in previous time slice
                    if data_time == time_slices[0]:
                        data_dif = 0
                        data_dif_big = 0
                        data_dif_fraq = 0
                    else:
                        data_load_prev = all_topics[len(all_topics) - (len(topic_at_time_slice) * len(topic_term))][4]
                        data_dif = data_load - data_load_prev
                        data_dif_fraq = safe_div(data_dif, data_load_prev)

                        data_dif_big = data_dif * 100000

                    data = [data_word, data_topic, data_time, data_time_no, data_load, data_dif_big, data_dif_fraq]
                    all_topics.append(data)

            print(f"Finished gathering data from time slice {time}\n")

        df_output = pd.DataFrame(all_topics, columns=["word", "topic", "time", "time_no", "load", "dif_e5", "dif_fraq"])

        return df_output

    def generate_topic_detail_tables(self):
        """
        Generate a list with a data frame for each topic, where rows denote a word and columns a time slice.
        :param files: Needed for the column names of the data frames
        :return: List of data frames for each topic
        """

        time_slices = self.files

        topic_df_list = []

        # Gather data for each words in each topic in each time slice

        _, topic_term, _, _, vocab = self.model.dtm_vis(self.doc_term_matrix, 0)

        for topic_id in range(len(topic_term)):

            # Create data frame with dummy column having the length of the vocab
            df_topic = pd.DataFrame([0] * len(vocab))

            # For each time slice
            for time_id in range(len(time_slices)):

                # Get all topic-word distributions for time slice i
                _, topic_term, _, _, vocab = self.model.dtm_vis(self.doc_term_matrix, time_id)

                # Topic-word distribution for one topic at time slice i
                topic_at_time_slice = topic_term[topic_id]

                df_topic[time_slices[time_id]] = topic_at_time_slice

            df_topic.index = vocab
            df_topic = df_topic.drop(columns=[0])
            df_topic["topic"] = topic_id

            print(f"Finished gathering data for topic {topic_id}")

            file_path = f"output/topics/topic_{topic_id}.csv"
            df_topic.to_csv(file_path)

            print(f"Topic detail data frame written to {file_path}")

            topic_df_list.append(df_topic)

        self.topic_df_list = topic_df_list

    # def write_topic_df_to_excel(self, file_path):
    #
    #     # Create a Pandas Excel writer using XlsxWriter as the engine.
    #     writer = pd.ExcelWriter(file_path, engine='xlsxwriter')
    #
    #     # Write each topic dataframe to a different worksheet.
    #     for i in range(len(self.topic_df_list)):
    #         self.topic_df_list[i].to_excel(writer, sheet_name=f'topic_{i}')
    #
    #         print(f"Topic {i} written to excel sheet")
    #
    #     # Close the Pandas Excel writer and output the Excel file.
    #     writer.save()
    #
    #     print(f"Topic dataframes written to excel file under {file_path}")

    def calculate_word_dif(self, folder_path):

        for topic_id in range(self.num_topics):
            df = pd.read_csv(folder_path + f"topic_{topic_id}.csv", index_col=0)

            for i in range(len(self.files) - 1):

                df[f"dif_{self.files[i+1]}"] = df[self.files[i+1]] - df[self.files[i]]

            output_file_path = folder_path + f"topic_dif_{topic_id}.csv"
            df.to_csv(output_file_path)

            print(f"Finished calculating differences and created file {output_file_path}")

    def construct_final_topic_data(self, folder_path):

        df_ref = pd.read_csv(folder_path + "topic_dif_0.csv", index_col=0)

        vocab = df_ref.index

        topic_list = []

        for topic_id in range(self.num_topics):

            df = pd.read_csv(folder_path + f"topic_dif_{topic_id}.csv", index_col=0)
            df.dropna()

            time_slices = self.files

            for word in vocab:
                # Account for the nan word
                try:
                    for i in range(len(time_slices)):
                        time = time_slices[i]

                        load = df.loc[word, time]

                        if i == 0:
                            dif = 0
                        else:
                            dif = df.loc[word, "dif_" + time]

                        data = [word, topic_id, time, i, load, dif]
                        topic_list.append(data)
                except:
                    print(f"Error at topic {topic_id} and word {word}")

            print(f"Finished reformating topic {topic_id}")

        df_output = pd.DataFrame(topic_list, columns=["word", "topic", "time", "time_no", "load", "dif"])

        df_output.to_csv(folder_path + "all_topics.csv")