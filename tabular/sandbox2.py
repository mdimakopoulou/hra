# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 14:37:00 2018

@author: t-madima
"""

import pickle
import os
import yaml

import click
import numpy as np

from tabular.ai import AI
from tabular.experiment import SoCExperiment
from environment.fruit_collection import FruitCollectionMini, FruitCollectionSmall

np.set_printoptions(suppress=True, precision=2)


dir_path = os.path.dirname(os.path.realpath(__file__))
config = os.path.join(dir_path, 'config.yaml')
with open(config, 'r') as f:
    params = yaml.safe_load(f)

# Override
params['use_gvf'] = True
params['nb_epochs'] = 200
params['learning_method'] = "max"
params['alpha'] = 1
params['final_alpha'] = 1

rng = np.random.RandomState(params['random_seed'])
env = FruitCollectionMini(rendering=False, game_length=300)
for mc_count in range(params['nb_experiments']):
    ai_list = []
    if not params['use_gvf']:
        for _ in range(env.nb_targets):
            fruit_ai = AI(nb_actions=env.nb_actions, init_q=params['init_q'], gamma=params['gamma'],
                          alpha=params['alpha'], learning_method=params['learning_method'], rng=rng)
            ai_list.append(fruit_ai)
    else:
        for _ in env.possible_fruits:
            gvf_ai = AI(nb_actions=env.nb_actions, init_q=params['init_q'], gamma=params['gamma'],
                        alpha=params['alpha'], learning_method=params['learning_method'], rng=rng)
            ai_list.append(gvf_ai)
    expt = SoCExperiment(ai_list=ai_list, env=env, aggregator_epsilon=params['aggregator_epsilon'],
                         aggregator_final_epsilon=params['aggregator_final_epsilon'],
                         aggregator_decay_steps=params['aggregator_decay_steps'],
                         aggregator_decay_start=params['aggregator_decay_start'], final_alpha=params['final_alpha'],
                         alpha_decay_steps=params['alpha_decay_steps'], alpha_decay_start=params['alpha_decay_start'],
                         epoch_size=params['epoch_size'], folder_name=params['folder_name'],
                         folder_location=params['folder_location'],
                         nb_eval_episodes=params['nb_eval_episodes'], use_gvf=params['use_gvf'], rng=rng)
    with open(expt.folder_name + '/config.yaml', 'w') as y:
        yaml.safe_dump(params, y)  # saving params for future reference
    expt.do_epochs(number=params['nb_epochs'])
