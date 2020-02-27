import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class Evaluator:

    def __init__(self, topic_dif_file_path, files):

        self.topic_dif_file_path = topic_dif_file_path
        self.files = files

        self.data_binary_lag_1_pred = []
        self.data_binary_lag_1_test = []
        self.data_real_lag_1_pred = []
        self.data_real_lag_1_test = []
        self.data_binary_lag_2_pred = []
        self.data_binary_lag_2_test = []
        self.data_real_lag_2_pred = []
        self.data_real_lag_2_test = []
        self.df_eval = None

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

        print(f"Length of evaluation data set for time slice lag 1: {len(self.data_real_lag_1_pred)}")

        print("Finished transforming topic dif data into evaluation data sets for time slice lag 1")

        self.df_eval = pd.DataFrame(
            {
                "real_lag_1_pred": self.data_real_lag_1_pred,
                "real_lag_1_test": self.data_real_lag_1_test,
                "binary_lag_1_pred": self.data_binary_lag_1_pred,
                "binary_lag_1_test": self.data_binary_lag_1_test
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
        for topic in self.model.print_topics():
            topic_id = topic[0]
            terms_raw = topic[1].split("+")
            terms = [x.split("*")[1] for x in terms_raw]  # get rid of probabilities
            terms = [x.replace("\"", "").replace(" ", "") for x in
                     terms[:n]]  # get rid of quotation marks and whitespaces
            out.append(terms)

        return pd.DataFrame(out)