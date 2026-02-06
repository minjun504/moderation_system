import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import optuna
import json

from content_moderation_system.features import build_vectoriser, csr_to_tensor
from content_moderation_system.config import PROCESSED_DATA_DIR, CONFIG_DIR
from content_moderation_system.modeling.torch_dataset import SieveData, BertDataset
from content_moderation_system.modeling.architecture import TierOneFilter, DistilBertRegressor
from content_moderation_system.modeling.utils import train_one_epoch, validate, collate_dense

def objective(trial,
              model_type, 
              df_train, 
              df_test, 
              input_feature, 
              target_feature, 
              batch_size: int = 32, 
              random_seed: int = 42
):
    torch.manual_seed(random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train =  df_train[input_feature].values
    X_test = df_test[input_feature].values
    y_train = df_train[target_feature].values
    y_test = df_test[target_feature].values

    threshold = 0.5
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

        vectorizer = build_vectoriser(ngram_range=ngram_range, max_features=max_features)
        X_train = vectorizer.fit_transform(X_train)
        X_test = vectorizer.transform(X_test)
        X_train = csr_to_tensor(X_train)
        X_test = csr_to_tensor(X_test)

        train_dataset = SieveData(X_train, torch.tensor(y_train).float())
        test_dataset = SieveData(X_test, torch.tensor(y_test).float())
        model = TierOneFilter(vocab_size=len(vectorizer.get_feature_names_out())).to(device)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_dense)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_dense)

    
    elif model_type == "BERT":
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
        train_dataset = BertDataset(X_train.tolist(), y_train)
        test_dataset = BertDataset(X_test.tolist(), y_test)

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

    study.optimize(
        lambda trial: objective(trial, model_type, df_train, df_test, input_feature, target_feature), 
        n_trials = n_trials
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
    results["sieve"] = run_study("sieve", df_train, df_test, INPUT_FEATURE, TARGET_FEATURE, 20)
    results["BERT"] = run_study("BERT", df_train, df_test, INPUT_FEATURE, TARGET_FEATURE, 20)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=4)