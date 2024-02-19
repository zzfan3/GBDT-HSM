import sys
import os
           
from dhg.models import HGNN

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models.GBDT import GBDTCatBoost
from core.models.GNN import GNN
from core.models.BGNN import BGNN
from core.scripts.utils1 import NpEncoder

import configs.config as config
import json
import time
import datetime
from pathlib import Path
from collections import defaultdict as ddict

import pandas as pd
import networkx as nx
import random
import numpy as np
import fire
from omegaconf import OmegaConf
from sklearn.model_selection import ParameterGrid

config._init()

class RunModel:
    def read_input(self, input_folder):
        self.X = pd.read_csv(f'{input_folder}/X.csv')
        self.y = pd.read_csv(f'{input_folder}/y.csv')

        networkx_graph = nx.read_graphml(f'{input_folder}/graph.graphml')
        networkx_graph = nx.relabel_nodes(networkx_graph, {str(i): i for i in range(len(networkx_graph))})
        self.networkx_graph = networkx_graph

        categorical_columns = []
        if os.path.exists(f'{input_folder}/cat_features.txt'):
            with open(f'{input_folder}/cat_features.txt') as f:
                for line in f:
                    if line.strip():
                        categorical_columns.append(line.strip())

        self.cat_features = None
        if categorical_columns:
            columns = self.X.columns
            self.cat_features = np.where(columns.isin(categorical_columns))[0]

            for col in list(columns[self.cat_features]):
                self.X[col] = self.X[col].astype(str)

        if os.path.exists(f'{input_folder}/masks.json'):
            with open(f'{input_folder}/masks.json') as f:
                self.masks = json.load(f)
        else:
            print('Creating and saving train/val/test masks')
            idx = list(range(self.y.shape[0]))
            self.masks = dict()
            for i in range(self.max_seeds):
                random.shuffle(idx)
                r1, r2, r3 = idx[:int(.6 * len(idx))], idx[int(.6 * len(idx)):int(.8 * len(idx))], idx[
                                                                                                   int(.8 * len(idx)):]
                self.masks[str(i)] = {"train": r1, "val": r2, "test": r3}

            with open(f'{input_folder}/masks.json', 'w+') as f:
                json.dump(self.masks, f, cls=NpEncoder)

    def get_input(self, dataset_dir, dataset: str):
        if dataset == 'vk_class':
            input_folder = dataset_dir / 'vk_class'
        elif dataset == 'house_class':
            input_folder = dataset_dir / 'house_class'
        elif dataset == 'dblp':
            input_folder = dataset_dir / 'dblp'
        elif dataset == 'slap':
            input_folder = dataset_dir / 'slap'
        elif dataset == 'NTU_mv_gv':
            input_folder = dataset_dir / 'NTU_gvmv'
        elif dataset == 'NTU_gvcnn':
            input_folder = dataset_dir / 'NTU_gvcnn'
        elif dataset == 'NTU_mvcnn':
            input_folder = dataset_dir / 'NTU_mvcnn'
        elif dataset == 'ModeNet40_gvcnn':
            input_folder = dataset_dir / 'ModeNet40_gvcnn'
        elif dataset == 'ModeNet40_mvcnn':
            input_folder = dataset_dir / 'ModeNet40_mvcnn'
        elif dataset == 'ModeNet40_gvmv':
            input_folder = dataset_dir / 'ModeNet40_gvmv'
        else:
            input_folder = dataset

        if self.save_folder is None:
            self.save_folder = f'results/{dataset}/{datetime.datetime.now().strftime("%d_%m")}'

        self.read_input(input_folder)
        print('Save to folder:', self.save_folder)

    def run_one_model(self, config_fn, model_name):
        self.config = OmegaConf.load(config_fn)
        grid = ParameterGrid(dict(self.config.hp))

        for ps in grid:
            param_string = ''.join([f'-{key}{ps[key]}' for key in ps])
            exp_name = f'{model_name}{param_string}'
            print(f'\nSeed {self.seed} RUNNING:{exp_name}')

            config.set("filename", exp_name + "_seed_" + self.seed)

            with open("loss.csv", "a+") as loss_f:
                loss_f.write("Seed " + str(self.seed) + " " + str(exp_name) + "\n")
            
            with open("macro.csv", "a+") as macro_f:
                macro_f.write("Seed " + str(self.seed) + " " + str(exp_name) + "\n")

            runs = []
            runs_custom = []
            times = []
            for _ in range(self.repeat_exp):
                start = time.time()
                model = self.define_model(model_name, ps)

                inputs = {'X': self.X, 'y': self.y, 'train_mask': self.train_mask,
                          'val_mask': self.val_mask, 'test_mask': self.test_mask, 'cat_features': self.cat_features}
                if model_name in ['gnn', 'resgnn', 'bgnn']:
                    inputs['networkx_graph'] = self.networkx_graph

                metrics = model.fit(num_epochs=self.config.num_epochs, patience=self.config.patience,
                                    loss_fn=f"{self.seed_folder}/{exp_name}.txt",
                                    metric_name='loss' if self.task == 'regression' else 'accuracy', **inputs)
                finish = time.time()
                best_loss = min(metrics['loss'], key=lambda x: x[1])
                best_custom = max(metrics['r2' if self.task == 'regression' else 'accuracy'], key=lambda x: x[1])
                runs.append(best_loss)
                runs_custom.append(best_custom)
                times.append(finish - start)
            self.store_results[exp_name] = (list(map(np.mean, zip(*runs))),
                                            list(map(np.mean, zip(*runs_custom))),
                                            np.mean(times),
                                            )

    def define_model(self, model_name, ps):
        if model_name == 'catboost':
            return GBDTCatBoost(self.task, **ps)
        elif model_name == 'gnn':
            return GNN(self.task, **ps)
        elif model_name == 'resgnn':
            gbdt = GBDTCatBoost(self.task)
            gbdt.fit(self.X, self.y, self.train_mask, self.val_mask, self.test_mask,
                     cat_features=self.cat_features,
                     num_epochs=1000, patience=100,
                     plot=False, verbose=False, loss_fn=None,
                     metric_name='loss' if self.task == 'regression' else 'accuracy')
            return GNN(task=self.task, gbdt_predictions=gbdt.model.predict(self.X), **ps)
        elif model_name == 'bgnn':
            return BGNN(self.task, **ps)
        elif model_name in ['GB_HGNN', 'GB_HGNNP', 'GB_HSM']:
            return BGNN(self.task, **ps)

    def create_save_folder(self, seed):
        self.seed_folder = f'{self.save_folder}/{seed}'
        os.makedirs(self.seed_folder, exist_ok=True)

    def split_masks(self, seed):
        self.train_mask, self.val_mask, self.test_mask = self.masks[seed]['train'], \
            self.masks[seed]['val'], self.masks[seed]['test']

    def save_results(self, seed):
        self.seed_results[seed] = self.store_results
        with open(f'{self.save_folder}/seed_results.json', 'w+') as f:
            json.dump(self.seed_results, f)

        self.aggregated = self.aggregate_results()
        with open(f'{self.save_folder}/aggregated_results.json', 'w+') as f:
            json.dump(self.aggregated, f)

    def get_model_name(self, exp_name: str, algos: list):
        # get name of the model (for gnn-like models (eg. gat))
        if 'name' in exp_name:
            model_name = '-' + [param[4:] for param in exp_name.split('-') if param.startswith('name')][0]
        else:
            model_name = ''

        # get a model used a MLP (eg. MLP-GNN)
        if 'gnn' in exp_name and 'mlpTrue' in exp_name:
            model_name += '-MLP'

        # algo corresponds to type of the model (eg. gnn, resgnn, bgnn)
        for algo in algos:
            if exp_name.startswith(algo):
                return algo + model_name
        return 'unknown'

    def aggregate_results(self):
        algos = ['catboost', 'gnn', 'resgnn', 'bgnn', 'GB_HGNN', 'GB_HGNNP', 'GB_HSM']
        model_best_score = ddict(list)
        model_best_time = ddict(list)

        results = self.seed_results
        for seed in results:
            model_results_for_seed = ddict(list)
            for name, output in results[seed].items():
                model_name = self.get_model_name(name, algos=algos)
                if self.task == 'regression':  # rmse metric
                    val_metric, test_metric, time = output[0][1], output[0][2], output[2]
                else:  # accuracy metric
                    val_metric, test_metric, time = output[1][1], output[1][2], output[2]
                model_results_for_seed[model_name].append((val_metric, test_metric, time))

            for model_name, model_results in model_results_for_seed.items():
                if self.task == 'regression':
                    best_result = min(model_results)  # rmse
                else:
                    best_result = max(model_results)  # accuracy
                model_best_score[model_name].append(best_result[1])
                model_best_time[model_name].append(best_result[2])

        aggregated = dict()
        for model, scores in model_best_score.items():
            aggregated[model] = (np.mean(scores), np.std(scores),
                                 np.mean(model_best_time[model]), np.std(model_best_time[model]))
        return aggregated

    def run(self, dataset: str, *args,
            save_folder: str = None,
            task: str = 'regression',
            repeat_exp: int = 1,
            max_seeds: int = 5,
            dataset_dir: str = None,
            config_dir: str = None
            ):
        start2run = time.time()
        self.repeat_exp = repeat_exp
        self.max_seeds = max_seeds
        print(dataset, args, task, repeat_exp, max_seeds, dataset_dir, config_dir)

        with open("loss.csv", "a+") as loss_f:
            loss_f.write(str(dataset) + "," + str(args) + "\n")
        
        with open("macro.csv", "a+") as macro_f:
            macro_f.write(str(dataset) + "," + str(args) + "\n")
        
        dataset_dir = Path(dataset_dir) if dataset_dir else Path(__file__).parent / 'datasets'
        config_dir = Path(config_dir) if config_dir else Path(__file__).parent / 'configs' / 'model'
        print(dataset_dir, config_dir)

        self.task = task
        self.save_folder = save_folder
        self.get_input(dataset_dir, dataset)

        self.seed_results = dict()
        for ix, seed in enumerate(self.masks):
            print(f'{dataset} Seed {seed}')
            config.set("dataset", dataset)
            
            self.seed = seed

            self.create_save_folder(seed)
            self.split_masks(seed)

            self.store_results = dict()
            for arg in args:
                if arg == 'all':
                    self.run_one_model(config_fn=config_dir / 'catboost.yaml', model_name="catboost")
                    self.run_one_model(config_fn=config_dir / 'gnn.yaml', model_name="gnn")
                    self.run_one_model(config_fn=config_dir / 'resgnn.yaml', model_name="resgnn")
                    self.run_one_model(config_fn=config_dir / 'bgnn.yaml', model_name="bgnn")
                    self.run_one_model(config_fn=config_dir / 'hgnn.yaml', model_name="gnn")
                    self.run_one_model(config_fn=config_dir / 'gb_hsm.yaml', model_name="bgnn")
                    break
                elif arg == 'catboost':
                    self.run_one_model(config_fn=config_dir / 'catboost.yaml', model_name="catboost")
                elif arg == 'gnn':
                    self.run_one_model(config_fn=config_dir / 'gnn.yaml', model_name="gnn")
                elif arg == 'hgnn':
                    self.run_one_model(config_fn=config_dir / 'hgnn.yaml', model_name="gnn")
                elif arg == 'resgnn':
                    self.run_one_model(config_fn=config_dir / 'resgnn.yaml', model_name="resgnn")
                elif arg == 'bgnn':
                    self.run_one_model(config_fn=config_dir / 'bgnn.yaml', model_name="bgnn")
                elif arg == 'gb_hsm':
                    self.run_one_model(config_fn=config_dir / 'gb_hsm.yaml', model_name="bgnn")
                
            self.save_results(seed)
            if ix + 1 >= max_seeds:
                break

        print(f'Finished {dataset}: {time.time() - start2run} sec.')


if __name__ == '__main__':
    RunModel().run('slap', 'gb_hsm', task='classification')
