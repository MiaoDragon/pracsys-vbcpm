"""
evaluate the task planner based on the following metrics:
- # actions vs. scene difficulty
- reduced occlusion vs. # actions taken
- how many samples to sample placement vs. # objects
"""
import pandas as pd
import numpy as np
import os

from pandas.core.frame import DataFrame
def performance_eval(actions, occlusions, planner_samples, success, scene_difficulty, fname):
    """
    fname: the filename of the csv file to record performance
    """
    dtypes = {}
    dtypes['# actions'] = 'int32'
    dtypes['samples per action'] = 'float64'
    dtypes['# objects'] = 'int32'
    dtypes['occlusion reduction ratio per action'] = 'float64'
    dtypes['success'] = 'int32'

    if os.path.exists(fname):
        df = pd.read_csv(fname, index_col=0)
        df = df.astype(dtypes)
    else:
        df = pd.DataFrame()
    object_n = scene_difficulty['object_n']
    new_data = {}
    new_data['# actions'] = len(actions)
    new_data['samples per action'] = np.sum(planner_samples) / len(actions)
    new_data['# objects'] = object_n
    new_data['occlusion reduction ratio per action'] = (occlusions[0][0].astype(int).sum() - occlusions[-1][1].astype(int).sum()) / occlusions[0][0].astype(int).sum() / len(actions)
    new_data['success'] = success
    new_df = DataFrame(new_data, index=[0]).astype(dtypes)
    df = df.append(new_df)
    df.to_csv(fname)