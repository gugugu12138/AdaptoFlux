# experiments/ExternalBaselines/baselines/gplearn_runner.py
import numpy as np
from gplearn.genetic import SymbolicRegressor
from ..utils import is_numerical_exact_match, train_test_split_feynman

def run_gplearn(X, y, test_size=0.5, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split_feynman(X, y, test_size=test_size, random_state=random_state)
    
    # experiments/ExternalBaselines/baselines/gplearn_runner.py
    est = SymbolicRegressor(
        population_size=1000,
        generations=20,
        stopping_criteria=1e-8,
        function_set=('add', 'sub', 'mul', 'div'),
        random_state=random_state
    )
    est.fit(X_train, y_train)
    
    y_pred = est.predict(X_test)
    mse = np.mean((y_pred - y_test) ** 2)
    exact = is_numerical_exact_match(y_pred, y_test)
    
    return {
        "exact_match": bool(exact),
        "mse": float(mse),
        "expression": str(est._program)
    }