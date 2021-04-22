import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import pickle
from enum import Enum


class AlgorithmChoice(Enum):
    DECISION_TREE_CLASSIFIER = 1
    RANDOM_FOREST_CLASSIFIER = 2


class Experiment:

    def __init__(self):
        return

    def execute(self, X, y, scoring, filename, algorithm=AlgorithmChoice.DECISION_TREE_CLASSIFIER, test_split=0.3):
        # dt = DecisionTreeClassifier(random_state=42)
        # rf = RandomForestClassifier(random_state=42)
        # nb = GaussianNB()
        # knn = KNeighborsClassifier()
        param_grid = {}
        alg = None
        # algorithm = [dt, rf]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_split)
        experiment_results = []
        # for alg in algorithm:
        if algorithm == AlgorithmChoice.DECISION_TREE_CLASSIFIER:
            alg = DecisionTreeClassifier(random_state=42)
            param_grid = {
                'max_depth': np.arange(1, 21),
                'criterion': ['gini', 'entropy'],
                'splitter': ['best', 'random'],
                'min_samples_split': np.arange(1, 21)
            }
        elif algorithm == AlgorithmChoice.RANDOM_FOREST_CLASSIFIER:
            alg = RandomForestClassifier(random_state=42)
            param_grid = {
                'n_estimators': [range(100, 500, 50)],
                'max_features': ['auto', 'sqrt', 'log2'],
                'max_depth': np.arange(1, 21),
                'criterion': ['gini', 'entropy'],
                'min_samples_split': np.arange(1, 21)
            }

        grid_search = GridSearchCV(alg, param_grid, cv=10)
        gs = grid_search.fit(X_train, y_train)
        model = gs.best_estimator_
        results = cross_validate(model, X_train, y_train, scoring=scoring)
        # model.fit(X_train, y_train)
        y_pred = model.predict(X_test, y_test)
        experiment_results.append([model, results, gs.best_params_, y_pred])

        pickle.dump(experiment_results, open(filename, 'wb'))
        filename.close()