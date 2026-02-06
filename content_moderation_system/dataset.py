import pandas as pd
from sklearn.model_selection import train_test_split
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR

def load_data(
        input_path,
        output_path,
        input_var: str, 
        target_var: str, 
        random_state: int=42, 
        test_size: float=0.2
):
    df = pd.read_csv(input_path)
    
    df = df.dropna(subset=[input_var, target_var])
    df = df.reset_index(drop=True)

    df["target_binned"] = pd.qcut(df[target_var], q=5, labels=False, duplicates="drop")
    df_train, df_test  = train_test_split(df,
                                          test_size=test_size,
                                          random_state=random_state,
                                          stratify=df["target_binned"])
    df_train = df_train.drop(columns=["target_binned"])
    df_test = df_test.drop(columns=["target_binned"])

    output_path.mkdir(parents=True, exist_ok=True)
    df_train.to_csv(output_path / "processed_train.csv", index=False)
    df_test.to_csv(output_path / "processed_test.csv", index=False)

if __name__=="__main__":
    load_data(RAW_DATA_DIR/"train.csv", PROCESSED_DATA_DIR, "comment_text", "target")
    
