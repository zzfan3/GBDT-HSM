from catboost import Pool, CatBoostClassifier, CatBoostRegressor
import time
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score, f1_score
import numpy as np
from collections import defaultdict as ddict

class GBDTCatBoost:
    def __init__(self, task='regression', depth=6, lr=0.1, l2_leaf_reg=None, max_bin=None):
        self.task = task
        self.depth = depth
        self.learning_rate = lr
        self.l2_leaf_reg = l2_leaf_reg
        self.max_bin = max_bin


    def init_model(self, num_epochs, patience):
        catboost_model_obj = CatBoostRegressor if self.task == 'regression' else CatBoostClassifier
        self.catboost_loss_function = 'RMSE' if self.task == 'regression' else 'MultiClass'
        self.custom_metrics = ['R2'] if self.task == 'regression' else ['Accuracy', 'TotalF1:average=Macro', 'MultiClass']
        # ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC', 'R2'],

        self.model = catboost_model_obj(iterations=num_epochs,
                                       depth=self.depth,
                                       learning_rate=self.learning_rate,
                                       loss_function=self.catboost_loss_function,
                                       custom_metric=self.custom_metrics,
                                       random_seed=0,
                                       early_stopping_rounds=patience,
                                       l2_leaf_reg=self.l2_leaf_reg,
                                       max_bin=self.max_bin,
                                       nan_mode='Min'
                                        )

    def get_metrics(self):
        d = self.model.evals_result_

        metrics = ddict(list)
        keys = ['learn', 'validation_0', 'validation_1'] \
            if 'validation_0' in self.model.evals_result_ \
            else ['learn', 'validation']
        for metric_name in d[keys[0]]:
            perf = [d[key][metric_name] for key in keys]
            
            if metric_name == self.catboost_loss_function:
                metrics['loss'] = list(zip(*perf))
            else:
                metrics[metric_name.lower()] = list(zip(*perf))
        return metrics

    def get_test_metric(self, metrics, metric_name, metric_name_addon = 'totalf1:average=macro'):
        if metric_name == 'loss':
            val_epoch = np.argmin([acc[1] for acc in metrics[metric_name]])
        else:
            val_epoch = np.argmax([acc[1] for acc in metrics[metric_name]])

        min_metric = metrics[metric_name][val_epoch]
        min_metric_addon = metrics[metric_name_addon][val_epoch]
        return min_metric, min_metric_addon, val_epoch

    def save_metrics(self, metrics, fn):
        with open(fn, "w+") as f:
            for key, value in metrics.items():
                print(key, value, file=f)

    def train_val_test_split(self, X, y, train_mask, val_mask, test_mask):
        X_train, y_train = X.iloc[train_mask], y.iloc[train_mask]
        X_val, y_val = X.iloc[val_mask], y.iloc[val_mask]
        X_test, y_test = X.iloc[test_mask], y.iloc[test_mask]
        return X_train, y_train, X_val, y_val, X_test, y_test

    def fit(self,
            X, y, train_mask, val_mask, test_mask,
            cat_features=None, num_epochs=1000, patience=200,
            plot=False, verbose=False,
            loss_fn="", metric_name='loss'):

        X_train, y_train, X_val, y_val, X_test, y_test = \
            self.train_val_test_split(X, y, train_mask, val_mask, test_mask)
        self.init_model(num_epochs, patience)

        start = time.time()
        pool = Pool(X_train, y_train, cat_features=cat_features)
        # 将多eval改 单个eval
        eval_set = [(X_val, y_val), (X_test, y_test)]

        self.model.fit(pool, eval_set=eval_set, plot=plot, verbose=verbose)
        finish = time.time()

        num_trees = self.model.tree_count_
        print('Finished training. Total time: {:.2f} | Number of trees: {:d} | Time per tree: {:.2f}'.format(finish - start, num_trees, (time.time() - start )/num_trees))

        metrics = self.get_metrics()
        min_metric, min_metric_addon, min_val_epoch = self.get_test_metric(metrics, metric_name)
        if loss_fn:
            self.save_metrics(metrics, loss_fn)
        print('Best {} at iteration {}: {:.4f}/{:.4f}/{:.4f} with addon {:.4f}/{:.4f}/{:.4f}'.format(metric_name, min_val_epoch, *min_metric, *min_metric_addon))
        return metrics

    def predict(self, X_test, y_test):
        pred = self.model.predict(X_test)

        metrics = {}
        metrics['rmse'] = mean_squared_error(pred, y_test) ** .5

        return metrics