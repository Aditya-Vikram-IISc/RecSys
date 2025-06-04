import pandas as pd


class DataUtils:
    @staticmethod
    def read_data_from_path(path, sep, column_names):
        # read the data
        df = pd.read_csv(path, sep= sep, header= None)

        df.columns = column_names
        return df
    
    @staticmethod
    def split_df_to_train_test(df, split_method, seed = 0, test_frac = 0.1):
        # Check for valid method
        if split_method not in ["randomized", "sequence_aware"]:
            raise ValueError(f"Got {split_method} as input for split_method, valid inputs are 'randomized', 'sequence_aware'")

        # For random, random split test fraction of data to test 
        if split_method == "randomized":
            df_len = df.shape[0]
            df = df.sample(frac=1).reset_index(drop = True)

            split_idx = int(df_len*test_frac)
            test_df = df.iloc[:split_idx]
            train_df = df.iloc[split_idx:]
            return train_df, test_df
        
        elif split_method == "sequence_aware":
            # Get the last record for each user in test_df
            df_sorted = df.sort_values(by=["user_id", "timestamp"])
            last_idx = df_sorted.groupby('user_id')['timestamp'].idxmax()

            # Create the two DataFrames
            test_df = df.loc[last_idx]
            train_df = df.drop(index=last_idx)
            return train_df, test_df
        
    @staticmethod
    def prepare_data(df):
        user_list = []
        item_list = []
        score_list = []
        for line in df.itertuples():
            user_list.append(line[1])
            item_list.append(line[2])
            score_list.append(line[3])

        return user_list, item_list, score_list