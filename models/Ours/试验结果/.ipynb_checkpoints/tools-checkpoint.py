# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from copy import deepcopy
import pandas as pd
from ay2.tools.pandas import format_numeric_of_df_columns
from IPython.display import HTML, display

# %%
ROOT_PATH = "/home/ay/data/DATA/1-model_save/00-Deepfake/1-df-audio-new"

# %%
TASKCLASSES = {
    "LibriSeVoc_inner": 'LibriseVocInnerEvaluation',
    "wavefake_inner": 'WaveFakeInnerEvaluation',
    "DECRO_english": 'DECROInnerEvaluation',
    "DECRO_chinese": 'DECROInnerEvaluation',
    "LibriSeVoc_cross_method": 'LibriseVocCrossMethodEvaluation', 
    "wavefake_cross_method":'WaveFakeCrossMethodEvaluation',
    "ASV2021_inner" : 'ASV2021CrossMethodEvaluation',
    "LibriSeVoc_cross_dataset" : 'LibriSeVocCrossDatasetEvaluation',
}


# %% [markdown]
# # Help Functions

# %%
def read_test_result_from_csv_file(model, task, version=0, metric_prefix="test", file_name="test"):
    """
    Read test results from a CSV file: `f"{ROOT_PATH}/{model}/{task}/version_{version}/{file_name}.csv"`

    Args:
        model (str): The name of the model.
        task (str): The task for which the results are being read.
        version (int, optional): The version number of the model. Defaults to 0.
        metric_prefix (str, optional): The prefix used for metric columns in the CSV file. Defaults to "test".
        file_name (str, optional): The name of the CSV file containing the test results. Defaults to "test".

    Returns:
        pd.DataFrame or None: A DataFrame containing the requested metrics, or None if the file does not exist.
    """
    save_path = f"{ROOT_PATH}/{model}/{task}/version_{version}"
    csv_path = os.path.join(save_path, f"{file_name}.csv")

    if not os.path.exists(csv_path):
        print("Warning!!!! cannot find: ", csv_path)
        return pd.DataFrame()

    data = pd.read_csv(csv_path)
    data = data[[f"{metric_prefix}-acc", f"{metric_prefix}-auc", f"{metric_prefix}-eer"]]
    data["model"] = model
    data["task"] = task
    data["version"] = version
    return data


# %%
def generate_res_column_for_df(data: pd.DataFrame, acc=1, auc=1, eer=1, metric_prefix="test") -> pd.DataFrame:
    """
    Generate a new column in the DataFrame containing formatted model metrics.

    Args:
        data (pd.DataFrame): The input DataFrame with columns for 'test-acc', 'test-auc', and 'test-eer'.
        acc (bool, optional): Include the accuracy metric in the output. Defaults to True.
        auc (bool, optional): Include the AUC metric in the output. Defaults to True.
        eer (bool, optional): Include the Equal Error Rate (EER) metric in the output. Defaults to True.
        metric_prefix (str, optional): The prefix used for metric columns. Defaults to "test".

    Returns:
        pd.DataFrame: The input DataFrame with an additional column 'res' containing formatted metrics.
    """

    def help_format(x):
        _acc = "{:.2f}".format(x[f"{metric_prefix}-acc"] * 100)
        _auc = "{:.2f}".format(x[f"{metric_prefix}-auc"] * 100)
        _eer = "{:.2f}".format(x[f"{metric_prefix}-eer"] * 100)

        res = []
        if acc:
            res.append(_acc)
        if auc:
            res.append(_auc)
        if eer:
            res.append(_eer)
        res = "/".join(res) or ""
        return res

    data['res'] = data.apply(
        lambda x: help_format(x),
        axis=1,
    )

    return data


# %%
@dataclass
class ReadTaskResult:
    model: str
    task: str
    version: Union[List[int], int]
    calc_avg_on_version: bool = True
    
    def __post_init__(self):
        try:
            if isinstance(self.version, int) or len(self.version) == 1:
                self.data = self.post_read(
                    read_test_result_from_csv_file(model=self.model, task=self.task, version=self.version)
                )
            else:
                self.data = [
                    self.post_read(read_test_result_from_csv_file(model=self.model, task=self.task, version=v))
                    for v in self.version
                ]
                if self.calc_avg_on_version:
                    self.data = self.deal_multi_versions(self.data)
                else:
                    DATA = []
                    for _df, _v in zip(self.data, self.version):
                        _df['model'] = f"{self.model}---v{'%02d'%_v}"
                        DATA.append(_df)
                    self.data = pd.concat(DATA, ignore_index=True)
        except ValueError as e:
            print(self.model, self.task, self.version, e)
            self.data = pd.DataFrame()

    def post_read(self, data):
        return data

    def deal_multi_versions(self, list_datas: List[pd.DataFrame]) -> NotImplementedError:
        raise NotImplementedError()

# %% [markdown]
# # Inner Evaluation

# %%
class InnerEvaluation(ReadTaskResult):
    def post_read(self, data):
        return data[0:1]

    def deal_multi_versions(self, list_datas: List[pd.DataFrame]) -> NotImplementedError:
        data = pd.concat(list_datas, ignore_index=True)

        data = data.mean(numeric_only=True)
        data = dict(data)
        data['model'] = self.model
        data['task'] = self.task
        return pd.DataFrame([data])

class WaveFakeInnerEvaluation(InnerEvaluation):
    task: str = "wavefake_inner"

class LibriseVocInnerEvaluation(InnerEvaluation):
    task: str = "LibriSeVoc_inner"

class DECROInnerEvaluation(InnerEvaluation):
    task: str = "DECRO_english"


# %%
class DisplayInnerEvaluation:
    def __init__(
        self,
        models: List[str],
        versions: List[Union[List[int], int]] = [],
        tasks: List[str] = None,
        calc_avg_on_version=True,
        acc=0,
        auc=1,
        eer=1,
        task_classes=TASKCLASSES,
    ):
        self.acc = acc
        self.auc = auc
        self.eer = eer
        self.models = models
        self.tasks = tasks
        self.versions = versions
        self.calc_avg_on_version = calc_avg_on_version
        self.task_classes = task_classes
        
        self.data = self.read_data_for_model_task()

    def read_data_for_model_task(self):
        DATA = []
        for i, model in enumerate(self.models):
            for task in self.tasks:
                # print(model, task, TASKCLASSES[task])
                data = eval(self.task_classes[task])(model, task, self.versions[i], calc_avg_on_version=self.calc_avg_on_version).data
                DATA.append(data)
        data = pd.concat(DATA, ignore_index=True)
        return data

    def sort_df_rows(self, data):
        models = list(data.index)
        flag = 0
        for x in models:
            if '---' in x:
                flag = 1
        if not flag:
            data = data.loc[[x for x in self.models if x in models]]
            return data
        
        df = data.copy()
        df['modelname'] = [x.split('---')[0] for x in models]
        df['version'] = [x.split('---')[1] if x.split('---')[1] else "0"  for x in models]
        df_sorted = df.sort_values(by=['modelname', 'version'], key=lambda x: x.map({name: idx for idx, name in enumerate(self.models)}))
        return df_sorted
    
    def display(self, show_latex=0):
        data = self.data
        avg_data = data.groupby('model').mean(numeric_only=True).reset_index()
        avg_data['task'] = 'avg'
        data = pd.concat([data, avg_data], ignore_index=True)
        
        data = generate_res_column_for_df(data, acc=self.acc, auc=self.auc, eer=self.eer)
        data = data.pivot(index="model", columns=["task"], values="res").reset_index(drop=False).set_index("model")
        
        # data = data.loc[models]

        data = self.sort_df_rows(data)
        
        data = data[self.tasks + ['avg']]
        if show_latex:
            res = data.style.to_latex(column_format="lrr")
            res = res.replace("_", "-").replace("$", "")
            print(res)
        return data


# %%
# _versions = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
# MODELs_VERSIONS = {
#     "LCNN": _versions,
#     "RawNet2": _versions,
#     "RawGAT": _versions,
#     "Wave2Vec2": _versions,
#     "WaveLM": _versions,
#     "LibriSeVoc": _versions,
#     "AudioClip": _versions,
#     "Wav2Clip": _versions,
#     "AASIST": _versions,
#     "SFATNet": _versions,
#     "ASDG": _versions,
#     "Ours/ResNet": _versions,
# }
# models = list(MODELs_VERSIONS.keys())
# versions = list(MODELs_VERSIONS.values())

# %%
# tasks = ["LibriSeVoc_inner", "wavefake_inner", "DECRO_english", "DECRO_chinese"]
# res = DisplayInnerEvaluation(models, versions=versions, tasks=tasks, acc=0, calc_avg_on_version=False)
# res.display(show_latex=0)

# %% [markdown]
# # Cross Method Evaluation

# %%
class CrossMethodEvaluation(ReadTaskResult):
    def post_read(self, data):
        raise NotImplementedError

    def deal_multi_versions(self, list_datas: List[pd.DataFrame]) -> NotImplementedError:
        data = pd.concat(list_datas, ignore_index=True)
        if self.model == "Wave2Vec2":
            print(data)
            data.to_csv("hello.csv")
        # data = data.groupby(['model', 'task', 'method']).mean(numeric_only=True).reset_index()
        data = data.groupby(['model', 'task', 'method']).agg(['mean', 'std'], numeric_only=True).reset_index()
        data.columns = ['_'.join(col).strip() for col in data.columns.values]
        return data
    
    def calcuate_avg_on_methods(self, res, _range=None):
        if _range is None:
            res.loc[len(res)] = res.mean(numeric_only=True).copy()
        else:
            res.loc[len(res)] = res.iloc[_range[0] : _range[1]].mean(numeric_only=True).copy()
        res['model'] = self.model
        res['task'] = self.task
        res['version'] = res.iloc[0]['version']
        return res.copy()

class WaveFakeCrossMethodEvaluation(CrossMethodEvaluation):
    def post_read(self, data):
        res = data.iloc[0:6].copy()
        res = self.calcuate_avg_on_methods(res)
        res["method"] = [2, 3, 4, 5, 6, 7, 999]
        return res

class LibriseVocCrossMethodEvaluation(CrossMethodEvaluation):
    def post_read(self, data):
        res = data.iloc[0:4].copy()
        res = self.calcuate_avg_on_methods(res)
        res["method"] = [1, 2, 3, 5, 999]
        return res


class ASV2021CrossMethodEvaluation(CrossMethodEvaluation):
    def post_read(self, data):
        res = data.copy()
        # res = self.calcuate_avg_on_methods(res)
        try:
            res["method"] = [
                "0inner",
                "1AR",
                "2Non-AR",
                "3traditional",
                "4unknown",
                "5concat",
                "6whole",
            ]
        except ValueError:
            print(self.model, data.iloc[0]['version'])
        return res

class DisplayCrossMethodEvaluation(DisplayInnerEvaluation):


    def display(self, show_latex=0):
        data = self.data
        
        data = generate_res_column_for_df(data, acc=self.acc, auc=self.auc, eer=self.eer)
        data = data.pivot(index="model", columns=["task", "method"], values="res").reset_index(drop=False).set_index("model")
        
        data = self.sort_df_rows(data)
        
        data = format_numeric_of_df_columns(data)
        if show_latex:
            res = data.style.to_latex(column_format="lrr")
            res = res.replace("_", "-").replace("$", "")
            print(res)
        return data


# %%
# tasks=["LibriSeVoc_cross_method", 'wavefake_cross_method']

# res = DisplayCrossMethodEvaluation(models, versions=versions, tasks=tasks, acc=0, calc_avg_on_version=False, task_classes=TASKCLASSES)
# res.display(show_latex=0)

# %%
# tasks=["ASV2021_inner"]
# res = DisplayCrossMethodEvaluation(models, versions=versions, tasks=tasks, acc=0, task_classes=TASKCLASSES, calc_avg_on_version=False)
# res.display(show_latex=0).to_csv('ASV.csv')

# %%
# tasks=["ASV2021_inner"]
# res = DisplayCrossMethodEvaluation(models, versions=[10]*len(models), tasks=tasks, acc=0, task_classes=TASKCLASSES)
# res.display(show_latex=0)

# %% [markdown]
# # Crossdataset Evaluation

# %%
class LibriSeVocCrossDatasetEvaluation(CrossMethodEvaluation):
    def post_read(self, data):
        res = data.copy()
        res = self.calcuate_avg_on_methods(res, _range=(0, 7))
        try:
            res["method"] = [0, 1, 2, 3, 4, 5, 6, "inTheWild", "en", "ch"] + ["avg"]
        except ValueError:
            print(self.model, data.iloc[0]['version'])
        return res


# %%
# res = DisplayCrossMethodEvaluation(models, versions=versions, tasks=['LibriSeVoc_cross_dataset'], acc=0)
# res.display(show_latex=0)

# %% [markdown]
# # Cross Language Evaluation

# %%
CROSS_LANGUAGE_TASKCLASSES = {
    "wavefake_cross_lang" : 'WavefakeCrossLanguageEvaluation',
    "DECRO_chinese" : 'DECROCrossLanguageEvaluation',
    "DECRO_english" : 'DECROCrossLanguageEvaluation',
}


# %%
class WavefakeCrossLanguageEvaluation(InnerEvaluation):
    def post_read(self, data):
        return data

class DECROCrossLanguageEvaluation(InnerEvaluation):
    def post_read(self, data):
        return data[1:2]

# %%
# tasks=["wavefake_cross_lang", "DECRO_chinese", "DECRO_english"]

# res = DisplayInnerEvaluation(models, versions=versions, tasks=tasks, acc=0, task_classes=CROSS_LANGUAGE_TASKCLASSES,calc_avg_on_version=False)
# res.display(show_latex=0)

# %%
# tasks=["wavefake_cross_lang", "DECRO_chinese", "DECRO_english"]

# res = DisplayInnerEvaluation(models, versions=versions, tasks=tasks, acc=0, task_classes=CROSS_LANGUAGE_TASKCLASSES,calc_avg_on_version=True)
# res.display(show_latex=0)
