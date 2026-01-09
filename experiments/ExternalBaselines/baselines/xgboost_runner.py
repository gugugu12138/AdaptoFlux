# experiments/ExternalBaselines/baselines/xgboost_runner.py
import numpy as np
from xgboost import XGBRegressor
from ..utils import is_numerical_exact_match, train_test_split_feynman

def run_xgboost(X, y, test_size=0.5, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split_feynman(X, y, test_size=test_size, random_state=random_state)
    
    model = XGBRegressor(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = np.mean((y_pred - y_test) ** 2)
    exact = is_numerical_exact_match(y_pred, y_test)
    
    return {
        "exact_match": bool(exact),
        "mse": float(mse)
    }