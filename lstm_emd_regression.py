import numpy as np
import pandas as pd

from PyEMD import CEEMDAN
import matplotlib.pyplot as plt
import torch
import lightning.pytorch as pl
from pytorch_forecasting import TimeSeriesDataSet, RecurrentNetwork
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE
from tft_model import read_csv
from pytorch_forecasting.data.encoders import NaNLabelEncoder
from pytorch_forecasting.models.xlstm import xLSTMTime
from sklearn.metrics import mean_absolute_error
from pandas.tseries.offsets import DateOffset

import warnings
warnings.filterwarnings("ignore")


def main():
    # --- 1. PREPARE DATA ---
    # Read your data (Assuming read_csv is defined as before)
    train_df, valid_df, test_df, train_df_all = read_csv()

    # Combine for EMD processing (we need continuity)
    full_data = train_df_all.copy()
    original_signal = full_data['Obs_F10_7'].to_numpy()

    # Add integer time index (required for Pytorch Forecasting)
    full_data['time_idx'] = np.arange(len(full_data))
    full_data['group_id'] = 'series_1'  # Required for single series

    # --- 2. PERFORM EMD ---
    # (Optional: Add Linear Padding here if edge effects persist, omitted for clarity)
    #imfs = emd.sift.sift(original_signal)
    #n_imfs = imfs.shape[1]


    ceemdan = CEEMDAN()
    imfs = ceemdan(original_signal)
    print(imfs.shape)
    n_imfs = imfs.shape[0]


    # Add IMFs back to dataframe
    imf_cols = []
    for i in range(n_imfs):
        col_name = f'imf_{i}'
        full_data[col_name] = imfs[i, :]
        imf_cols.append(col_name)

    # Ensure external regressors are float32
    regressors = ["cycle_sin", "cycle_cos"]  # Add others like 'solar_rotation' if you have them
    for reg in regressors:
        full_data[reg] = full_data[reg].astype(np.float32)

    # --- 3. DEFINE TRAINING LOOP FOR EACH IMF ---
    horizon = 7
    encoder_length = 30  # Lookback window
    all_forecasts = []

    cutoff_2 = full_data["timestamp"].quantile(0.8)  # or a specific date
    train_df = full_data[(full_data["timestamp"] <= cutoff_2)]
    valid_df = full_data[(full_data["timestamp"] > cutoff_2)]

    print(f"Training LSTMs on {n_imfs} IMFs...")

    for i, target_imf in enumerate(imf_cols):
        print(f"--> Training model for {target_imf}...")
        target = target_imf
        max_prediction_length = 7
        max_encoder_length = 30
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
    #        target_normalizer= GroupNormalizer(groups=["group_ids"], transformation="softplus", center=True),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )

        # Create DataLoaders
        batch_size = 128
        train_dataloader = training_dataset.to_dataloader(train=True, batch_size=batch_size, num_workers=0)

        # Validation dataset (technically the last window of training data here for simplicity)
        validation = TimeSeriesDataSet.from_dataset(training_dataset, valid_df, predict=False, stop_randomization=True)
        val_dataloader = validation.to_dataloader(train=False, batch_size=256, num_workers=0)

        # Define LSTM Model
        # RNN with LSTM cell
        net = xLSTMTime.from_dataset(
            training_dataset,
            input_size = 29,
            output_size = 1,
            learning_rate=0.008,
            # Architecture hyperparameters
            hidden_size=64, # Embedding dimension
            dropout=0.03,
            loss=MAE(),
        )

        # Train with PyTorch Lightning
        trainer = pl.Trainer(
            max_epochs=10,  # Adjust based on your needs (IMFs converge fast)
            accelerator="auto",
            gradient_clip_val=0.1,
            num_sanity_val_steps = 0,
            enable_model_summary=True,
            enable_checkpointing=False,
            logger=True
        )

        trainer.fit(
            net,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )

        # --- PREDICT ---
        # Get the LAST valid prediction window (which corresponds to the horizon)
        best_model = net
        raw_predictions= best_model.predict(val_dataloader, mode="raw", return_x=True)

        # Extract just the prediction values (shape: [batch, horizon])
        # We take the last entry because we want the forecast starting from end of training
        imf_pred = raw_predictions.output.prediction.cpu().numpy()

        # Handle case where model predicts quantiles (take median) or mean
        if imf_pred.ndim > 1:
            imf_pred = imf_pred.flatten()  # Simplification for point forecast

        all_forecasts.append(imf_pred)

    # --- 4. AGGREGATE AND PLOT ---
    # Sum up all IMF predictions
    final_forecast = np.sum(np.stack(all_forecasts), axis=0)

    # Get Ground Truth
    ground_truth = valid_df['Obs_F10_7'].to_numpy()
    history_vals = train_df_all['Obs_F10_7'].to_numpy()
    history_dates = train_df_all['timestamp']

    validation_dates = valid_df['timestamp']
    # Create date index for prediction

    last_date = pd.to_datetime(history_dates.iloc[-1])
    future_dates = [last_date + DateOffset(days=x + 1) for x in range(len(valid_df) + 6)]

    plt.figure(figsize=(12, 6))
    plt.plot(history_dates, history_vals, label='History', color='#1f77b4')
    plt.plot(validation_dates, ground_truth, label='Actual', color='green', linewidth=2, marker='o')
    plt.plot(future_dates, final_forecast, label='EMD-LSTM Prediction', color='red', linewidth=3, marker='x')

    plt.title("EMD + LSTM (Pytorch Forecasting) with Regressors")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("lstmemd_actual_vs_predicted_all.png", dpi=300, bbox_inches="tight")


    print("MAE:", mean_absolute_error(ground_truth, final_forecast[:len(ground_truth)]))
    return final_forecast

if __name__ == "__main__":
    final_forecast = main()
