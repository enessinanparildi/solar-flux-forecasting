import numpy as np

from tft_model import read_csv, add_cycle_features
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric

# This simulates: "Train up to May 2024, predict 7 days.
# Then add the actual data from that week, retrain, and predict the next 7 days."
import emd
from statsmodels.tsa.ar_model import AutoReg
from sklearn.linear_model import LinearRegression


if __name__ == "__main__":

    run_cv = False
    train_df, valid_df, test_df, train_df_all = read_csv()
    train_df_all.rename(columns={'timestamp':'ds'}, inplace=True)
    test_df.rename(columns={'timestamp':'ds'}, inplace=True)
    imf_train = emd.sift.sift(train_df_all['Obs_F10_7'].to_numpy())
    test_df['y'] = test_df['Obs_F10_7']

    all_forecasts = []
    all_forecasts_upper = []
    all_forecasts_lower = []
    horizon = 7

    final_forecast_auto_reg = np.zeros(horizon)

    n_imfs = imf_train.shape[1]
    fig, axes = plt.subplots(n_imfs, 1, figsize=(10, 3 * n_imfs), sharex=True)

    for imf_ind in range(imf_train.shape[1]):

        train_df_all['y'] = imf_train[:, imf_ind]

        regressors = ["yy", "mm", "dd", "cycle_sin", "cycle_cos"]
        m = Prophet(seasonality_mode='additive', changepoint_prior_scale  = 0.8, daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True)
        m.add_seasonality(name='solar_rotation', period=27, fourier_order=5)
        m.add_seasonality(name='weekly', period=7, fourier_order=3, prior_scale=10)

        m.add_regressor("cycle_cos", mode='additive', prior_scale=50.0)
        m.add_regressor("cycle_sin", mode='additive', prior_scale=50.0)

        m.fit(train_df_all)

        if run_cv:
            # This takes 4 hours.
            df_cv = cross_validation(
                m,
                initial='1825 days',  # How much training data to start with
                period='7 days',  # How often to retrain
                horizon='7 days'  # How far to predict
            )
            df_p = performance_metrics(df_cv)
            df_cv.to_csv("prophet_cv_results.csv")


            plt.figure(figsize=(10, 6))
            plt.plot(df_cv['ds'], df_cv['y'], label='Actual')
            plt.plot(df_cv['ds'], df_cv['yhat'], label='Predicted (Rolling)')
            plt.legend()
            plt.savefig("prophet_actual_vs_predicted_all_cv.png", dpi=300, bbox_inches="tight")

        future = m.make_future_dataframe(periods=7, include_history=False)
        future.rename(columns={'ds':'timestamp'}, inplace=True)
        future = add_cycle_features(future)
        future.rename(columns={'timestamp':'ds'}, inplace=True)
        forecast = m.predict(future)
        all_forecasts.append(forecast['yhat'].to_numpy())
        all_forecasts_upper.append(forecast['yhat_lower'].to_numpy())
        all_forecasts_lower.append(forecast['yhat_upper'].to_numpy())

        current_imf = imf_train[:, imf_ind]

        if imf_ind == (n_imfs - 1) or imf_ind == (n_imfs - 2):
            # Fit a simple line on the last 50 days of the Trend IMF
            X_trend = np.arange(len(current_imf) - 50, len(current_imf)).reshape(-1, 1)
            y_trend = current_imf[-50:]

            lin_reg = LinearRegression().fit(X_trend, y_trend)

            # Predict forward
            X_future = np.arange(len(current_imf), len(current_imf) + horizon).reshape(-1, 1)
            pred = lin_reg.predict(X_future)
        else:

            trend_type = 't' if imf_ind == (n_imfs - 1) else 'c'  # Use linear trend for the last IMF only

            model = AutoReg(current_imf, lags=365, trend="n").fit()

            # Predict future steps
            # start index is len(data), end index is len(data) + horizon - 1
            pred = model.predict(start=len(current_imf), end=len(current_imf) + horizon - 1)

        # Add to total
        final_forecast_auto_reg += pred

        axes[imf_ind].plot(current_imf[-400:], label='History')
        axes[imf_ind].plot(np.arange(400, 400 + horizon), pred, label='Pred', color='red')
        axes[imf_ind].set_title(f"IMF {imf_ind}")

    plt.tight_layout()
    plt.savefig("imf_preds.png", dpi=300, bbox_inches="tight")

    # 4. Final Plot
    plt.figure(figsize=(12, 6))
    # Plot last 100 days of actuals
    plt.plot(range(100), train_df_all['Obs_F10_7'].iloc[-100:], label='Actual History')
    # Plot Ground Truth (Test set)
    plt.plot(np.arange(100, 100 + horizon), test_df['Obs_F10_7'].iloc[-8:-1], label='Ground Truth (Test)',
             color='green')
    # Plot Prediction
    plt.plot(np.arange(100, 100 + horizon), final_forecast_auto_reg, label='EMD-AR Prediction', color='red', linewidth=2)

    plt.legend()
    plt.title("EMD + AutoRegression Results")
    plt.savefig("imf_preds_auto_reg.png", dpi=300, bbox_inches="tight")

    print(mean_absolute_error(test_df['y'].iloc[-8:-1], final_forecast_auto_reg))


    final_forecast = np.sum(np.stack(all_forecasts), axis=0)
    final_forecast_lower = np.sum(np.stack(all_forecasts_lower), axis=0)
    final_forecast_upper = np.sum(np.stack(all_forecasts_upper), axis=0)

    plt.figure(figsize=(10, 5))
    plt.plot(test_df['ds'], test_df['y'], label='Actual')
    plt.plot(forecast['ds'], final_forecast, label='Predicted')
    plt.fill_between(
        forecast['ds'],
        final_forecast_lower,
        final_forecast_upper,
        alpha=0.3
    )

    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.title('Prophet: Actual vs Predicted')
    plt.tight_layout()

    plt.savefig("prophet_actual_vs_predicted.png", dpi=300, bbox_inches="tight")

    predict_all  = False
    if predict_all:
        future = m.make_future_dataframe(periods=len(test_df), include_history=False)
        future.rename(columns={'ds':'timestamp'}, inplace=True)
        future = add_cycle_features(future)
        future.rename(columns={'timestamp':'ds'}, inplace=True)
        forecast = m.predict(future)

        plt.figure(figsize=(10, 5))
        plt.plot(test_df['ds'], test_df['y'], label='Actual')
        plt.plot(forecast['ds'], final_forecast, label='Predicted')
        plt.fill_between(
            forecast['ds'],
            forecast['yhat_lower'],
            forecast['yhat_upper'],
            alpha=0.3
        )

        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.title('Prophet: Actual vs Predicted')
        plt.tight_layout()

        plt.savefig("prophet_actual_vs_predicted_all.png", dpi=300, bbox_inches="tight")