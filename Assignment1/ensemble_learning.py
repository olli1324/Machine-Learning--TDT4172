import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_log_error

class EnsembleLearning:
    def __init__(self):
        self.model = None

    def load_data(self, train_path='final_mission_train.csv', test_path='final_mission_test.csv'):
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        return train, test

    def shift_data(self, test):
        test_shifted = test.copy()
        test_shifted.iloc[:, 1:-1] = test.iloc[:, :-2].values
        test_shifted['nexus_rating'] = test.iloc[:, -1]
        test_shifted['grid_connections'] = test.iloc[:, -2]
        return test_shifted

    def prepare_data(self, train, test_shifted):
        target = 'nexus_rating'
        y_train = train[target]
        y_test = test_shifted[target]
        X_train = train.drop(columns=[target])
        X_test = test_shifted.drop(columns=[target])
        y_train = np.log1p(y_train)
        return X_train, X_test, y_train, y_test

    def fit(self, X_train, y_train):
        self.model = CatBoostRegressor(iterations=1000, learning_rate=0.1, depth=6, l2_leaf_reg=10, random_state=69)
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        y_pred = self.model.predict(X_test)
        y_pred = np.expm1(y_pred)
        return y_pred

    def evaluate(self, y_test, y_pred):
        rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
        return rmsle
    
    

def main():
    EL = EnsembleLearning()
    train, test = EL.load_data()
    test_shifted = EL.shift_data(test)
    X_train, X_test, y_train, y_test = EL.prepare_data(train, test_shifted)
    EL.fit(X_train, y_train)
    y_pred = EL.predict(X_test)
    rmsle = EL.evaluate(y_test, y_pred)
    print(f"RMSLE: {rmsle:.4f}")
    

if __name__ == '__main__':
    main()