import optuna
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import numpy as np


print("Loading seasonal data...")

summer_data = pd.read_feather('data/summer_data.feather')
autumn_data = pd.read_feather('data/autumn_data.feather')
winter_data = pd.read_feather('data/winter_data.feather')
spring_data = pd.read_feather('data/spring_data.feather')

data_list = [summer_data, autumn_data, winter_data, spring_data]
pred_season = ['autumn', 'winter', 'spring', 'summer']


def objective(trial):

    if trial.number >= 100:
        study.stop()

    params = {
        "objective": "reg:squarederror",
        "max_depth": trial.suggest_int("max_depth", 6, 36),
        "learning_rate": trial.suggest_float("learning_rate", 1e-1, 0.9),
        "lambda": trial.suggest_float("reg_lambda", 1e-2, 0.9),
        "alpha": trial.suggest_float("reg_alpha", 1e-2, 0.9),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
    }

    pruning_callback = optuna.integration.XGBoostPruningCallback(
        trial, "validation-rmse")

    bst = xgb.train(params, Dtrain, 100,
                    evals=[(Dtest, "validation")],
                    callbacks=[pruning_callback],
                    verbose_eval=False)

    preds = bst.predict(Dval)
    error = metrics.mean_squared_error(y_test, preds)

    return error


for i, data in enumerate(data_list):

    print(f"Training for {pred_season[i]}")

    # Normalize current data
    # NOTE: Decision trees are indifferent to normalization, unlike NNs.
    # Normalization code will be removed in the future.
    x_scaler = MinMaxScaler()
    x_scaler.fit(data.loc[:, 'temp850':'prgpcp'])

    y_scaler = MinMaxScaler()
    y_scaler.fit(data.iloc[:, -1].to_numpy().reshape(-1, 1))

    # Split train and test set
    trainval_data = data.loc[data['year'] < 2017]
    test_data = data.loc[data['year'] >= 2017]

    # Prepare training data
    X = x_scaler.transform(trainval_data.loc[:, 'temp850':'prgpcp'].values)
    y = y_scaler.transform(trainval_data.iloc[:, -1].values.reshape(-1, 1))

    # Prepare test data
    test_data_input = test_data.loc[:, 'temp850':'prgpcp']
    test_data_x = x_scaler.transform(test_data_input)
    test_data_output = test_data.iloc[:, -1]
    test_data_y = y_scaler.transform(test_data_output.values.reshape(-1, 1))

    # Split training and validation data from the training dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.75,
                                                        shuffle=True)

    # Create DMatrices (data type for XGBoost)
    Dtrain = xgb.DMatrix(X_train, label=y_train)
    Dval = xgb.DMatrix(X_test, label=y_test)
    Dtest = xgb.DMatrix(test_data_x, label=test_data_y)
    DX = xgb.DMatrix(X, label=y)

    # Optimize
    study_name = 'season-' + pred_season[i]
    storage_name = "sqlite:///xgboost.db"

    study = optuna.create_study(direction="minimize",
                                pruner=optuna.pruners.HyperbandPruner(),
                                study_name=study_name,
                                storage=storage_name,
                                load_if_exists=True)
    study.optimize(objective, n_trials=100)

    # Save all trials
    df = study.trials_dataframe()
    df.to_excel("results/best-params-" + pred_season[i] + ".xlsx")

    # Retrain the model using the best hyperparameters and more rounds
    print("Retraining with 5000 trees...")
    bst = xgb.train(study.best_params, DX, 5000)

    # Do the inference in the test data and save the results
    output = bst.predict(Dtest)
    print("Error: ", metrics.mean_squared_error(test_data_y, output))
    np.save("xgboost-output-" + pred_season[i] + ".npy", output)
