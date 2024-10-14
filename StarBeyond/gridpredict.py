import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import optuna
# Load your CSV data
data = pd.read_csv('gridmachinelearning.csv')

# Define features and target
X = data[['params_DROPOUT', 'params_EPS', 'params_KERNEL_1', 'params_KERNEL_2', 
          'params_KERNEL_3', 'params_LR', 'params_OUT_CHANNELS_1', 
          'params_OUT_CHANNELS_2', 'params_OUT_CHANNELS_3', 'params_PADDING_1', 
          'params_PADDING_2', 'params_PADDING_3', 'params_RNNLAYER', 
          'params_RNNSIZE', 'params_STRIDE_1', 'params_STRIDE_2', 
          'params_STRIDE_3', 'params_WD', 'params_hidden1', 'params_hidden2', 
          'params_hidden3', 'params_poolsize1', 'params_poolsize2', 
          'params_poolsize3']]

y = data['value']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

def objective(trial):
    # Define the hyperparameter search space
    params = {
        'params_DROPOUT': trial.suggest_float('params_DROPOUT', 0.0, 0.5),
        'params_EPS': trial.suggest_loguniform('params_EPS', 1e-8, 1e-2),
        'params_KERNEL_1': trial.suggest_int('params_KERNEL_1', 10, 70),
        'params_KERNEL_2': trial.suggest_int('params_KERNEL_2', 10, 60),
        'params_KERNEL_3': trial.suggest_int('params_KERNEL_3', 5, 20),
        'params_LR': trial.suggest_loguniform('params_LR', 1e-5, 1e-2),
        'params_OUT_CHANNELS_1': trial.suggest_int('params_OUT_CHANNELS_1', 10, 128),
        'params_OUT_CHANNELS_2': trial.suggest_int('params_OUT_CHANNELS_2', 10, 128),
        'params_OUT_CHANNELS_3': trial.suggest_int('params_OUT_CHANNELS_3', 10, 128),
        'params_PADDING_1': trial.suggest_int('params_PADDING_1', 0, 5),
        'params_PADDING_2': trial.suggest_int('params_PADDING_2', 0, 5),
        'params_PADDING_3': trial.suggest_int('params_PADDING_3', 0, 5),
        'params_RNNLAYER': trial.suggest_int('params_RNNLAYER', 1, 3),
        'params_RNNSIZE': trial.suggest_int('params_RNNSIZE', 32, 512),
        'params_STRIDE_1': trial.suggest_int('params_STRIDE_1', 1, 3),
        'params_STRIDE_2': trial.suggest_int('params_STRIDE_2', 1, 3),
        'params_STRIDE_3': trial.suggest_int('params_STRIDE_3', 1, 3),
        'params_WD': trial.suggest_loguniform('params_WD', 1e-5, 1e-1),
        'params_hidden1': trial.suggest_int('params_hidden1', 10, 512),
        'params_hidden2': trial.suggest_int('params_hidden2', 10, 512),
        'params_hidden3': trial.suggest_int('params_hidden3', 10, 512),
        'params_poolsize1': trial.suggest_int('params_poolsize1', 2, 5),
        'params_poolsize2': trial.suggest_int('params_poolsize2', 2, 5),
        'params_poolsize3': trial.suggest_int('params_poolsize3', 2, 5)
    }
    
    # Predict the "number value" using the surrogate model
    X_trial = pd.DataFrame([params.values()], columns=params.keys())
    predicted_value = model.predict(X_trial)[0]
    
    # Objective is to minimize the difference to the target value (e.g., 0.004)
    target_value = 0.004
    return abs(predicted_value - target_value)

# Run the optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=10000)

# Print the best found hyperparameters
print("Best hyperparameters:")
for key, value in study.best_params.items():
    print(f"{key}: {value}")