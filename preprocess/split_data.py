

def train_test_split(df, train_ratio=0.8):
    train_size = int(len(df) * train_ratio)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    return train_df, test_df

def train_eval_test_split(df, train_ratio=0.8, eval_ratio=0.1):
    train_size = int(len(df) * train_ratio)
    eval_size = int(len(df) * eval_ratio)
    train_df = df.iloc[:train_size]
    eval_df = df.iloc[train_size:train_size+eval_size]
    test_df = df.iloc[train_size+eval_size:]
    return train_df, eval_df, test_df