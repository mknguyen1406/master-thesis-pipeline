import os
import pandas as pd


class DataAggregator:

    def __init__(self, directory=None, files=None, file_suffix="", file_format="", sheet_name="", df_array=None,
                 date_col=None, target_cols=None, dropna_cols=None):

        self.directory = directory
        self.files = files
        self.file_suffix = file_suffix
        self.file_format = file_format
        self.sheet_name = sheet_name
        self.df_array = df_array
        self.date_col = date_col
        self.target_cols = target_cols
        self.dropna_cols = dropna_cols

    def aggregate_data(self):

        if (self.directory is not None) & (self.files is not None):

            # merge all files in directory
            self.df_array = []

            for i in range(len(self.files)):

                # Check format
                if self.file_format == "csv":
                    if self.date_col:
                        df_file = pd.read_csv(self.directory + self.files[i] + self.file_suffix, index_col=0,
                                              parse_dates=[self.date_col])
                    else:
                        df_file = pd.read_csv(self.directory + self.files[i] + self.file_suffix)
                else:
                    df_file = pd.read_excel(self.directory + self.files[i] + self.file_suffix, index_col=0,
                                            sheet_name=self.sheet_name, parse_dates=[self.date_col])

                # Sort by date
                if self.date_col:
                    df_file = df_file.sort_values(self.date_col).reset_index(drop=True)

                # Drop NA rows for specified columns
                if self.dropna_cols:
                    df_file = df_file.dropna(subset=self.dropna_cols).reset_index(drop=True)

                # Add framework programme information
                if self.date_col:
                    df_file["fp"] = i + 1

                self.df_array.append(df_file)

        if self.df_array:

            df = self.df_array[0]
            for i in range(len(self.df_array) - 1):
                df = df.append(self.df_array[i + 1], ignore_index=True)

            # Select target columns
            if self.target_cols:
                df = df[self.target_cols]

            print(f"Aggregation completed")

            return df

    def save_to_csv(self, df):

        file_path = self.directory + "all_projects.csv"

        # Save to CSV
        df.to_csv(file_path)

        print(f"Aggregated file saved to {file_path}")

    def read_final_from_csv(self):

        file_path = self.directory + "all_projects.csv"

        # Read from csv
        df = pd.read_csv(file_path, index_col=0, parse_dates=[self.date_col])

        print(f"Aggregated file read from {file_path}")

        return df
