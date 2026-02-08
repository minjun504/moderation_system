import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import optuna
import json
import gc

from content_moderation_system.config import PROCESSED_DATA_DIR, CONFIG_DIR, MODELS_DIR
from content_moderation_system.modeling.torch_dataset import VectorDataset
from content_moderation_system.modeling.architecture import TierOneFilter, DistilBertRegressor
from content_moderation_system.modeling.utils import csr_to_tensor, train_one_epoch, validate, get_embeddings, collate_dense

def objective(trial,
              model_type=None, 
              df_train=None, 
              df_test=None, 
              encoded_train=None, 
              encoded_test=None, 
              input_feature="comment_text", 
              target_feature="target", 
              batch_size: int = 32, 
              random_seed: int = 42
):
    torch.manual_seed(random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train =  df_train[input_feature].values
    X_test = df_test[input_feature].values
    y_train = df_train[target_feature].values
    y_test = df_test[target_feature].values

    y_train_tensor = torch.tensor(y_train).float()
    y_test_tensor = torch.tensor(y_test).float()

    threshold = trial.suggest_float("imbalance_threshold", 0.1, 0.9)
    binary_labels = torch.tensor(y_train >= threshold).float()
    num_pos = binary_labels.sum()
    num_neg = len(binary_labels) - num_pos
    pos_weight = torch.tensor([num_neg / num_pos]).to(device)

    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)

    if model_type == "sieve":
        ngram_range = trial.suggest_categorical("ngram_range", ["1,1", "1,2"])
        if ngram_range == "1,1":
            ngram_range = (1, 1)
        else:
            ngram_range = (1, 2)
        max_features = trial.suggest_int("max_features", 5000, 20000)

        vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features)
        X_train = vectorizer.fit_transform(X_train)
        X_test = vectorizer.transform(X_test)
        X_train = csr_to_tensor(X_train)
        X_test = csr_to_tensor(X_test)

        train_dataset = VectorDataset(X_train, y_train_tensor)
        test_dataset = VectorDataset(X_test, y_test_tensor)
        model = TierOneFilter(vocab_size=len(vectorizer.get_feature_names_out())).to(device)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_dense)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_dense)
    
    elif model_type == "BERT":
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)

        train_dataset = VectorDataset(encoded_train, y_train_tensor)
        test_dataset = VectorDataset(encoded_test, y_test_tensor)
        model = DistilBertRegressor(dropout_rate=dropout_rate).to(device)
    
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    loss_fxn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

    epochs = 5
    for epoch in range(epochs):
        _ = train_one_epoch(model, train_loader, optimizer, loss_fxn, device)
        val_loss = validate(model, test_loader, loss_fxn, device)
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_loss

def run_study(
        model_type, 
        df_train, 
        df_test, 
        input_feature, 
        target_feature, 
        n_trials=20
):
    study = optuna.create_study(direction="minimize")

    if model_type == "sieve":
        study.optimize(
            lambda trial: objective(
                trial, "sieve",
                df_train=df_train, 
                df_test=df_test, 
                input_feature=input_feature, 
                target_feature=target_feature
            ), 
            n_trials=n_trials
        )

    if model_type == "BERT":
        encoder_path = MODELS_DIR / "quantized_encoder"
        
        train_texts = df_train[input_feature].tolist()
        embeddings_train = get_embeddings(train_texts, encoder_path).half()
        del train_texts

        test_texts = df_test[input_feature].tolist()
        embeddings_test = get_embeddings(test_texts, encoder_path).half()
        del test_texts
        gc.collect()

        study.optimize(
                    lambda trial: objective(
                        trial, model_type="BERT", 
                        df_train=df_train, df_test=df_test, 
                        encoded_train=embeddings_train, 
                        encoded_test=embeddings_test,
                        input_feature=input_feature, 
                        target_feature=target_feature
                    ), 
                    n_trials=n_trials
                )
        return study.best_params

if __name__ == "__main__":
    TRAIN_PATH = PROCESSED_DATA_DIR/"processed_train.csv"
    TEST_PATH = PROCESSED_DATA_DIR/"processed_test.csv"
    INPUT_FEATURE = "comment_text"
    TARGET_FEATURE = "target"
    OUTPUT_PATH = CONFIG_DIR/"best_hyperparams.json"

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df_train = pd.read_csv(TRAIN_PATH).sample(frac=0.1, random_state=42)
    df_test = pd.read_csv(TEST_PATH).sample(frac=0.1, random_state=42)

    results = {}
    results["sieve"] = run_study("sieve", df_train.copy(), df_test.copy(), INPUT_FEATURE, TARGET_FEATURE, 20)
    results["BERT"] = run_study("BERT", df_train, df_test, INPUT_FEATURE, TARGET_FEATURE, 20)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=4)