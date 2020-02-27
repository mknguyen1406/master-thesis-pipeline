import pandas as pd
from nltk.probability import FreqDist
from collections import Counter


class Explorer:

    def __init__(self, data_clean):

        self.data_clean = data_clean
        self.doc_count = len(data_clean)
        self.word_list = [word for doc in self.data_clean for word in doc]
        self.word_count = len(self.word_list)
        self.fdist = FreqDist(self.word_list)
        self.num_words = self.fdist.N()
        self.word_counts = self.fdist.most_common(self.num_words)

        self.df_words = None

    def calculate_word_df(self):

        columns = ["word", "count"]
        data = []

        for word, count in self.word_counts:

            # Create data sample
            data_record = [word, count]

            # Append to data
            data.append(data_record)

        # Create data frame
        df_words = pd.DataFrame(data, columns=columns)

        # Count no of docs that contain a word for each word
        result = Counter()
        for doc in self.data_clean:
            result.update(set(doc))

        # Create data frame
        df_doc_count = pd.DataFrame({
            "word": list(result.keys()),
            "doc_count": list(result.values())
        })

        # Merge data frames
        df_merged = df_words.merge(df_doc_count, on="word", how="left").reset_index(drop=True)

        # Calculate doc frac
        df_merged["doc_frac"] = df_merged["doc_count"]/self.doc_count

        self.df_words = df_merged

    def get_n_most_frequent_words(self, n):
        df = self.df_words.sort_values("count", ascending=False)
        return df[:n]

    def get_most_frequent_words_by_frac(self, x):
        df = self.df_words.sort_values("doc_count", ascending=False)
        return df[df["doc_frac"] >= x]

    def get_most_frequent_words_by_doc_count(self, c):
        df = self.df_words.sort_values("doc_count", ascending=False)
        return df[df["doc_count"] >= c]

    def get_n_least_frequent_words(self, n):
        df = self.df_words.sort_values("count", ascending=True)
        return df[:n]

    def get_least_frequent_words_by_frac(self, x):
        df = self.df_words.sort_values("doc_count", ascending=True)
        return df[df["doc_frac"] <= x]

    def get_least_frequent_words_by_doc_count(self, c):
        df = self.df_words.sort_values("doc_count", ascending=True)
        return df[df["doc_count"] <= c]
