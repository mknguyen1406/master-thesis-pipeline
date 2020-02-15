import pandas as pd
import numpy as np

from gensim.models.wrappers import DtmModel as DtmModelClass

import pickle # to save data to files for later use
import datetime

from gensim.test.utils import common_corpus, common_dictionary


class DtmModel:

    def __init__(self, date_col, path_to_dtm_binary, dictionary, doc_term_matrix, seed, num_topics, output_path):

        self.date_col = date_col
        self.path_to_dtm_binary = path_to_dtm_binary
        self.dictionary = dictionary
        self.doc_term_matrix = doc_term_matrix
        self.seed = seed
        self.num_topics = num_topics
        self.output_path = output_path

        self.time_slice_labels = None
        self.time_slices = None
        self.model = None

    def prepare_data(self, df):

        # Add year column to data frame
        def get_year(x):
            return x.year

        yrs = df[self.date_col].apply(lambda x: get_year(x))
        df["year"] = yrs

        # Get time slice labels
        self.time_slice_labels = df["year"].unique()
        self.time_slices = df.groupby("year").size()

        return df

    def train_model(self):

        # train DTM model
        print("Start time: {}".format(datetime.datetime.now()))

        self.model = DtmModelClass(
            self.path_to_dtm_binary, corpus=self.doc_term_matrix, id2word=self.dictionary,
            time_slices=self.time_slices,
            num_topics=20,
            rng_seed=0
        )

        print("End time: {}".format(datetime.datetime.now()))

    def save_model(self):

        # Save to file
        self.model.save(self.output_path)

    def top_term_table(self, topic, slices, topn=10):
        """Returns a dataframe with the top n terms in the topic for each of
        the given time slices."""

        data = {}
        data["Topic_ID"] = [topic] * topn
        data["Word_Rank"] = [i for i in range(topn)]
        for time_slice in slices:
            time = np.where(self.time_slice_labels == time_slice)[0][0]
            data[time_slice] = [
                term for p, term
                in self.model.show_topic(topic, time=time, topn=topn)
            ]
        df = pd.DataFrame(data)
        return df