import os
from collections import defaultdict

import mlflow
import mlflow.pytorch

import lightning.pytorch as pl
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
import torch

from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet, QuantileLoss
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.data.encoders import NaNLabelEncoder
from pytorch_forecasting.metrics import MAE, RMSE

from statsmodels.stats.diagnostic import het_arch
from statsmodels.tsa.seasonal import seasonal_decompose

from torch.utils.data import DataLoader

torch.set_float32_matmul_precision('medium')

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
matplotlib.use("Agg")  # disables GUI plotting


def add_cycle_features(df):
    cycle_starts = pd.to_datetime([
        '1933-09-01', '1944-02-01', '1954-04-01', '1964-10-01',
        '1976-06-01', '1986-09-01', '1996-05-01', '2008-12-01', '2019-12-01',
    ])

    ts = pd.to_datetime(df['timestamp'].values)
    no_cycle = ts < cycle_starts[0]
    cycle_idx = np.searchsorted(cycle_starts, ts, side='right') - 1
    cycle_idx = np.clip(cycle_idx, 0, len(cycle_starts) - 1)

    elapsed_days = np.array((ts - cycle_starts[cycle_idx]).days, dtype=float)
    years_elapsed = elapsed_days / 365.25
    years_elapsed[no_cycle] = 0.0

    df['years_since_min'] = years_elapsed
    df['cycle_sin'] = np.sin(2 * np.pi * years_elapsed / 11.0)
    df['cycle_cos'] = np.cos(2 * np.pi * years_elapsed / 11.0)
    df['cycle_num'] = (17 + cycle_idx).astype(str)
    df.loc[no_cycle, 'cycle_num'] = '16'
    return df


def plot_seasonal_decompose(df):
    s = df[["timestamp", "Obs_F10_7"]]

    s = np.log1p(s['Obs_F10_7'].astype(float))
    trs = seasonal_decompose(s, model='additive', period=365)

    # Check for heteroskedasticity
    resid = trs.resid
    resid = resid[~resid.isna()]
    test_stat, p_value, _, _ = het_arch(resid)

    # if < 0.05 Residuals are heteroskedastic (ARCH effect exists)
    print("ARCH test statistic:", test_stat)
    print("p-value:", p_value)

    fig = trs.plot()
    fig.set_size_inches(12, 8)
    fig.tight_layout()
    fig.savefig(f"non_log_seasonal_decompose.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_time_series(df):
    fig, ax = plt.subplots(figsize=(12, 4))
    df = df.iloc[-1000:]  # first 100 days

    s = np.log1p(df['Obs_F10_7'].astype(float))
    ax.plot(df['timestamp'], s )
    ax.set_title(f"solar over time")
    ax.set_xlabel("Date")
    ax.set_ylabel("solar")

    fig.savefig(f"plot_{1}.png", dpi=300, bbox_inches="tight")

def plot_corr_mat(df: pd.DataFrame) -> None:
    plt.figure(figsize=(50, 40))
    sns.heatmap(df.corr(), annot=True, fmt=".2f")
    plt.tight_layout()

    plt.savefig("plots/correlation_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

def read_feast(repo_path: str = "feature_repo") -> tuple:
    """Load training data from the Feast offline store.

    Drop-in replacement for read_csv(). Returns identical
    (train_df, valid_df, test_df, train_df_all) tuple but reads from the
    feature store instead of a raw file, eliminating training-serving skew.

    Prerequisites:
        python feature_repo/ingest.py   # populate offline store
        cd feature_repo && feast apply  # register feature definitions
    """
    from feast import FeatureStore

    store = FeatureStore(repo_path=repo_path)

    # Pull the full history from the offline store
    end   = pd.Timestamp.utcnow()
    start = pd.Timestamp("1932-01-01", tz="UTC")

    entity_df = pd.DataFrame({
        "data_source":    ["NOAA"],
        "event_timestamp": [end],
    })

    df = store.get_historical_features(
        entity_df=entity_df,
        features=store.get_feature_service("solar_training_features"),
    ).to_df()

    # Restore column names expected by wrap_datasets
    rename = {
        "obs_f10_7": "Obs_F10_7", "nd": "ND", "cp": "Cp", "c9": "C9",
        **{f"kp_{i}_{i+3}": f"Kp_{i}_{i+3}" for i in range(0, 24, 3)},
        **{f"ap_{i}_{i+3}": f"Ap_{i}_{i+3}" for i in range(0, 24, 3)},
    }
    df = df.rename(columns=rename)
    df["timestamp"] = pd.to_datetime(df["event_timestamp"]).dt.tz_localize(None)
    df["time_idx"]  = np.arange(len(df))
    df["group_ids"] = 1

    cutoff   = pd.to_datetime("2024-05-01")
    start_dt = cutoff - pd.Timedelta(days=60)

    train_df_all = df[df["timestamp"] < cutoff]
    cutoff_2     = train_df_all["timestamp"].quantile(0.8)
    train_df     = train_df_all[train_df_all["timestamp"] <= cutoff_2]
    valid_df     = train_df_all[train_df_all["timestamp"] >  cutoff_2]
    test_df      = df[df["timestamp"] >= start_dt].iloc[:68]

    return train_df, valid_df, test_df, train_df_all


def read_csv():
    with open("data/SW-All.txt", "r") as f:
        lines = f.readlines()
    end_num = 0
    begin_num = 0
    for num, line in enumerate(lines):
        if line == "END OBSERVED\n":
            end_num = num

        if line == "BEGIN OBSERVED\n":
            begin_num = num

    line_data = lines[begin_num + 1: end_num]


    cols = ["yy", "mm", "dd", "BSRN", "ND",
            "Kp","Kp","Kp","Kp","Kp","Kp","Kp","Kp",
            "Sum","Ap","Ap","Ap","Ap","Ap","Ap","Ap","Ap",
            "Avg","Cp","C9","ISN","Adj_F10_7","Q","Adj_Ctr81","Adj_Lst81",
            "Obs_F10_7","Obs_Ctr81","Obs_Lst81"]

    # Generate 3-hour interval suffixes
    suffixes = [f"{i}_{i+3}" for i in range(0, 24, 3)]

    out_cols = []
    kp_idx = 0
    ap_idx = 0

    for c in cols:
        if c == "Kp":
            out_cols.append(f"Kp_{suffixes[kp_idx]}")
            kp_idx += 1
        elif c == "Ap":
            out_cols.append(f"Ap_{suffixes[ap_idx]}")
            ap_idx += 1
        else:
            out_cols.append(c)

    data_dict = defaultdict(list)
    for line in line_data:
        row_data= line[:-1].split(" ")
        row_data = [data for data in row_data if len(data) > 0]
        for ind, data in enumerate(row_data):
            data_dict[out_cols[ind]].append(data)

    df = pd.DataFrame(data_dict)
    for col in out_cols:
        df[col] = df[col].astype(float)

    df["timestamp"] = pd.to_datetime(
        df[["yy", "mm", "dd"]].rename(columns={"yy": "year", "mm": "month", "dd": "day"})
    )
    df['time_idx'] = np.array(list(df.index))
    df["group_ids"] = 1
    df['flux_lag_27'] = df['Obs_F10_7'].shift(27)
    df['flux_lag_27'] = df['flux_lag_27'].fillna(method='bfill')
    df = add_cycle_features(df)
    plot_corr_mat(df)

    cutoff = pd.to_datetime("2024-05-01")
    start = cutoff - pd.Timedelta(days=60)

    # or a specific date
    train_df_all = df[(df["timestamp"] < cutoff)]

    cutoff_2 = train_df_all["timestamp"].quantile(0.8) # or a specific date
    train_df = train_df_all[(train_df_all["timestamp"] <= cutoff_2)]
    valid_df = train_df_all[(train_df_all["timestamp"] > cutoff_2)]

    test_df = df[(df["timestamp"] >= start)].iloc[:68]
    return train_df, valid_df, test_df, train_df_all

def wrap_datasets(train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame,
                  train_df_all: object) -> tuple[DataLoader, DataLoader, DataLoader, TimeSeriesDataSet]:
    target = "Obs_F10_7"
    max_prediction_length = 7
    max_encoder_length = 60
    kp_features = ['Kp_0_3', 'Kp_3_6', 'Kp_6_9', 'Kp_9_12', 'Kp_12_15', 'Kp_15_18','Kp_18_21', 'Kp_21_24']
    ap_features = ['Ap_0_3', 'Ap_3_6', 'Ap_6_9', 'Ap_9_12', 'Ap_12_15', 'Ap_15_18', 'Ap_18_21', 'Ap_21_24']
    training_dataset = TimeSeriesDataSet(
        train_df,
        group_ids=["group_ids"],
        time_idx="time_idx",
        target=target,
        min_encoder_length=max_encoder_length// 2,  # keep encoder length long (as it is in the validation set)
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        categorical_encoders={
            "cycle_num": NaNLabelEncoder(add_nan=True)
        },
        static_categoricals=['cycle_num'],
        time_varying_known_reals=["yy", "mm", "dd", "cycle_sin", "cycle_cos"],
        time_varying_unknown_reals=["flux_lag_27", 'ND','Cp','C9'] + kp_features + ap_features,
        target_normalizer= GroupNormalizer(groups=["group_ids"], transformation="softplus", center=True),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    # create validation set (predict=True) which means to predict the last max_prediction_length points in time
    # for each series
    validation = TimeSeriesDataSet.from_dataset(
        training_dataset, valid_df, predict=False, stop_randomization=True,

    )
    test_data = TimeSeriesDataSet.from_dataset(training_dataset, test_df, predict=True, stop_randomization=True)

    batch_size = 128  # set this between 32 to 128
    train_dataloader = training_dataset.to_dataloader(
        train=True, batch_size=batch_size, num_workers=8, persistent_workers = True
    )
    val_dataloader = validation.to_dataloader(
        train=False, batch_size=batch_size * 10, num_workers=8, persistent_workers = True,
    )
    test_dataloader = test_data.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=8, persistent_workers = True)
    return train_dataloader, val_dataloader, test_dataloader, training_dataset

def get_trainer(trainer, tft, train_dataloader, val_dataloader):
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
    return best_tft, best_model_path

def get_module(train_dataloader, val_dataloader, training_dataset):
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=15, verbose=False, mode="min"
    )
    lr_logger = LearningRateMonitor()  # log the learning rate

    trainer = pl.Trainer(
        max_epochs=60,
        accelerator="gpu",
        precision="16-mixed",
        enable_model_summary=True,
        gradient_clip_val=0.01,
        #       limit_train_batches=50,  # coment in for training, running valiation every 30 batches
        # fast_dev_run=True,  # comment in to check that networkor dataset has no serious bugs
        callbacks=[lr_logger, early_stop_callback],
    )

    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=0.02,
        hidden_size=256,
        attention_head_size=16,
        dropout=0.05,
        hidden_continuous_size=64,
        loss=QuantileLoss(),
        log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
        optimizer="ranger",
        reduce_on_plateau_patience=4,
    )
    print(f"Number of parameters in network: {tft.size() / 1e3:.1f}k")
    return trainer, tft


def run_evaluation(best_tft, val_dataloader):

    predictions = best_tft.predict(
        val_dataloader, return_y=True, trainer_kwargs=dict(accelerator="cpu")
    )
    print(MAE()(predictions.output, predictions.y))

    raw_predictions = best_tft.predict(
        val_dataloader, mode="raw", return_x=True, trainer_kwargs=dict(accelerator="cpu")
    )
    #
    # raw_predictions.x['encoder_target'] = torch.squeeze(raw_predictions.x['encoder_target'])
    # raw_predictions.x['decoder_target'] = torch.squeeze(raw_predictions.x['decoder_target'])

    n_samples = len(raw_predictions.x['encoder_target'])

    for idx in range(n_samples):  # plot 10 examples
        fig = best_tft.plot_prediction(
            raw_predictions.x, raw_predictions.output, idx=idx, add_loss_to_title=True
        )
        fig.savefig(f"prediction_{idx}.png", dpi=300, bbox_inches="tight")

    predictions_vs_actuals = best_tft.calculate_prediction_actual_by_variable(
        predictions.x, predictions.output
    )

    fig_2 = best_tft.plot_prediction_actual_by_variable(predictions_vs_actuals)
    fig_2.savefig(f"prediction_plot_2.png", dpi=300, bbox_inches="tight")

    return predictions, raw_predictions

def get_dataloaders():
    train_df, valid_df, test_df, train_df_all = read_csv()
    train_dataloader, val_dataloader, test_dataloader, training_dataset = wrap_datasets(train_df, valid_df, test_df, train_df_all)
    return train_dataloader, val_dataloader, test_dataloader, training_dataset

def main():
    train_model = False

    train_df, valid_df, test_df, train_df_all = read_csv()

    plot_seasonal_decompose(train_df)
    plot_time_series(train_df)

    train_dataloader, val_dataloader, test_dataloader, training_dataset = wrap_datasets(train_df, valid_df, test_df,
                                                                                        train_df_all)
    torch.save(training_dataset.get_parameters(), "models/dataset_params.pt")
    trainer, tft = get_module(train_dataloader, val_dataloader, training_dataset)

    mlflow.set_experiment("solar_flux_tft")
    with mlflow.start_run():
        mlflow.log_params({
            "learning_rate": 0.02,
            "hidden_size": 256,
            "attention_head_size": 16,
            "dropout": 0.05,
            "hidden_continuous_size": 64,
            "max_epochs": 60,
            "batch_size": 128,
            "max_encoder_length": 60,
            "max_prediction_length": 7,
            "optimizer": "ranger",
            "precision": "16-mixed",
            "gradient_clip_val": 0.01,
            "train_model": train_model,
        })

        if train_model:
            best_tft, best_model_path = get_trainer(trainer, tft)
            mlflow.log_artifact(best_model_path, artifact_path="checkpoints")
        else:
            best_model_dir = '/perceptivespace/lightning_logs/version_171/checkpoints/epoch=29-step=2850.ckpt'
            best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_dir)
            mlflow.log_param("checkpoint", best_model_dir)

        mlflow.log_artifact("dataset_params.pt", artifact_path="dataset")

        # imf = emd.sift.sift(train_df['Obs_F10_7'].to_numpy())

        predictions = best_tft.predict(
            val_dataloader, return_y=True, return_x=True, trainer_kwargs=dict(accelerator="cpu")
        )
        test_predictions = best_tft.predict(
            test_dataloader, return_y=True, return_x=True, trainer_kwargs=dict(accelerator="cpu")
        )

        val_mae = (MAE()(predictions.output, predictions.y[0]) / predictions.output.shape[0]).item()
        test_mae = MAE()(test_predictions.output, test_predictions.y[0]).item()
        mlflow.log_metrics({"val_mae": val_mae, "test_mae": test_mae})
        print(val_mae)
        print(test_mae)

        raw_predictions = best_tft.predict(
            val_dataloader, mode="raw", return_x=True, trainer_kwargs=dict(accelerator="cpu")
        )
        raw_test_predictions = best_tft.predict(
            test_dataloader, mode="raw", return_y=True, return_x=True, trainer_kwargs=dict(accelerator="cpu")
        )
        pred_q = tft.predict(test_dataloader, mode="quantiles")

        n_samples = len(raw_predictions.x['encoder_target'])

        for idx in range(0, n_samples, 250):
            fig = best_tft.plot_prediction(
                raw_predictions.x, raw_predictions.output, idx=idx, add_loss_to_title=True
            )
            path = f"prediction_{idx}.png"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            mlflow.log_artifact(path, artifact_path="plots/val")

        for idx in range(1):
            fig = best_tft.plot_prediction(
                raw_test_predictions.x, raw_test_predictions.output, idx=idx, add_loss_to_title=True,
            )
            path = f"test_prediction_{idx}.png"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            mlflow.log_artifact(path, artifact_path="plots/test")

        predictions_vs_actuals = best_tft.calculate_prediction_actual_by_variable(
            predictions.x, predictions.output
        )

        fig_2 = best_tft.plot_prediction_actual_by_variable(predictions_vs_actuals)
        for key, value in fig_2.items():
            path = f"prediction_plot_{key}.png"
            value.savefig(path, dpi=300, bbox_inches="tight")
            mlflow.log_artifact(path, artifact_path="plots/feature_importance")


if __name__ == "__main__":
    main()