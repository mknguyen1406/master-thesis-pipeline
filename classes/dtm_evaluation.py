import pandas as pd
from gensim.models.coherencemodel import CoherenceModel

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class Evaluator:

    def __init__(self, topic_dif_file_path, files, model, num_time_slices, doc_term_matrix, dictionary,
                 coherence_metric='u_mass'):

        self.model = model
        self.num_time_slices = num_time_slices
        self.topic_dif_file_path = topic_dif_file_path
        self.files = files
        self.doc_term_matrix = doc_term_matrix
        self.dictionary = dictionary
        self.metric = coherence_metric

        self.data_binary_lag_1_pred = []
        self.data_binary_lag_1_test = []
        self.data_real_lag_1_pred = []
        self.data_real_lag_1_test = []
        self.data_real_dif_lag_1_pred = []
        self.data_real_dif_lag_1_test = []
        self.data_binary_lag_2_pred = []
        self.data_binary_lag_2_test = []
        self.data_real_lag_2_pred = []
        self.data_real_lag_2_test = []

        self.df_eval = None
        self.df_dif_eval = None

    def transform_topic_dif_data_lag_1(self):

        df_topic_detail = pd.read_csv(self.topic_dif_file_path, sep=";", decimal=",")

        print(f"Topic dif data read from {self.topic_dif_file_path}")

        # Construct evaluation data sets for 1 time slice lag
        for i in range(len(self.files) - 2):
            # current time slice
            time_now = "dif_" + self.files[i + 1]
            # Next time slice
            time_next = "dif_" + self.files[i + 2]

            # Construct data sets
            self.data_real_lag_1_pred.extend(df_topic_detail[time_now])
            self.data_real_lag_1_test.extend(df_topic_detail[time_next])

            self.data_binary_lag_1_pred = [int(dif > 0) for dif in self.data_real_lag_1_pred]
            self.data_binary_lag_1_test = [int(dif > 0) for dif in self.data_real_lag_1_test]

            if i > 0:
                # current time slice
                dif_time_now = "dif_dif_" + self.files[i + 1]
                # Next time slice
                dif_time_next = "dif_dif_" + self.files[i + 2]

                self.data_real_dif_lag_1_pred.extend(df_topic_detail[dif_time_now])
                self.data_real_dif_lag_1_test.extend(df_topic_detail[dif_time_next])

        print(f"Length of evaluation data set for time slice lag 1: {len(self.data_real_lag_1_pred)}")

        print("Finished transforming topic dif data into evaluation data sets for time slice lag 1")

        self.df_eval = pd.DataFrame(
            {
                "real_lag_1_pred": self.data_real_lag_1_pred,
                "real_lag_1_test": self.data_real_lag_1_test,
                "binary_lag_1_pred": self.data_binary_lag_1_pred,
                "binary_lag_1_test": self.data_binary_lag_1_test
            })

        self.df_dif_eval = pd.DataFrame(
            {
                "real_dif_lag_1_pred": self.data_real_dif_lag_1_pred,
                "real_dif_lag_1_test": self.data_real_dif_lag_1_test,
            })

        self.df_eval["binary_lag_1_tp_tn"] = self.df_eval["binary_lag_1_pred"] == self.df_eval["binary_lag_1_test"]

    def transform_topic_dif_data_lag_2(self):

        df_topic_detail = pd.read_csv(self.topic_dif_file_path, sep=";", decimal=",")

        print(f"Topic dif data read from {self.topic_dif_file_path}")

        # Construct evaluation data sets for 1 time slice lag
        for i in range(len(self.files) - 3):
            # current time slice
            time_now = "dif_" + self.files[i + 2]

            # Next time slice
            time_next = "dif_" + self.files[i + 3]

            # Construct data sets
            self.data_real_lag_2_pred.extend(df_topic_detail[time_now])
            self.data_real_lag_2_test.extend(df_topic_detail[time_next])

            self.data_binary_lag_2_pred = [int(dif > 0) for dif in self.data_real_lag_2_pred]
            self.data_binary_lag_2_test = [int(dif > 0) for dif in self.data_real_lag_2_test]

        print(f"Length of evaluation data set for time slice lag 2: {len(self.data_real_lag_2_pred)}")

        print("Finished transforming topic dif data into evaluation data sets for time slice lag 2")

        self.df_eval = pd.DataFrame(
            {
                "real_lag_2_pred": self.data_real_lag_2_pred,
                "real_lag_2_test": self.data_real_lag_2_test,
                "binary_lag_2_pred": self.data_binary_lag_2_pred,
                "binary_lag_2_test": self.data_binary_lag_2_test
            })

        self.df_eval["binary_lag_2_tp_tn"] = self.df_eval["binary_lag_2_pred"] == self.df_eval["binary_lag_2_test"]

    # print all topics
    def get_top_n_terms(self, n):
        out = []
        for topic_id in range(len(self.model.show_topics(num_topics=-1, times=8, num_words=n))):
            row = self.model.show_topics(num_topics=-1, times=8, num_words=n)[topic_id]
            row_words = row.split("+")
            row_words_cleaned = [x.split("*")[1] for x in row_words]  # get rid of probabilities
            row_words_cleaned = [x.replace("\"", "").replace(" ", "") for x in
                                 row_words_cleaned]  # get rid of probabilities
            out.append(row_words_cleaned)

        return pd.DataFrame(out)

    def get_coherence_over_time(self):

        coherences = []

        for time_slice in range(self.num_time_slices):

            # we just have to specify the time-slice we want to find coherence for.
            topics_wrapper = self.model.dtm_coherence(time=0)

            # running u_mass coherence on our models
            cm_wrapper = CoherenceModel(
                topics=topics_wrapper,
                corpus=self.doc_term_matrix,
                dictionary=self.dictionary,
                coherence=self.metric)

            # Calculate coherence
            coherence = cm_wrapper.get_coherence()
            coherences.append(coherence)

            print(f"{self.metric} coherence score at time {time_slice} is: \n{coherence}")

        return pd.DataFrame(coherences)

        # # to use 'c_v' we need texts, which we have saved to disk.
        # texts = pickle.load(open('Corpus/texts', 'rb'))
        # cm_wrapper = CoherenceModel(topics=topics_wrapper, texts=texts, dictionary=dictionary, coherence='c_v')
        # cm_DTM = CoherenceModel(topics=topics_dtm, texts=texts, dictionary=dictionary, coherence='c_v')
        #
        # print("C_v topic coherence")
        # print("Wrapper coherence is ", cm_wrapper.get_coherence())
        # print("DTM Python coherence is", cm_DTM.get_coherence())
