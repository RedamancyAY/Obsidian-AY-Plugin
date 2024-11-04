# +
import os

import pandas as pd
from IPython.display import HTML, display


# -


def format_res(x, acc=1, auc=1, eer=1, metric_prefix="test"):
    _acc = "${:5.2f}$".format(x[f"{metric_prefix}-acc"] * 100).replace(" ", "\phantom{0}")
    _auc = "${:5.2f}$".format(x[f"{metric_prefix}-auc"] * 100).replace(" ", "\phantom{0}")
    _eer = "${:5.2f}$".format(x[f"{metric_prefix}-eer"] * 100).replace(" ", "\phantom{0}")
    res = ""
    if acc:
        res += _acc
    if auc:
        if res:
            res += " / "
        res += _auc
    if eer:
        if res:
            res += " / "
        res += _eer
    return res


ROOT_PATH = "/home/ay/data/DATA/1-model_save/00-Deepfake/1-df-audio"


def read_test_result(model, task, version=0, metric_prefix="test", file_name="test"):
    save_path = f"{ROOT_PATH}/{model}/{task}/version_{version}"
    csv_path = os.path.join(save_path, f"{file_name}.csv")

    if not os.path.exists(csv_path):
        print("Warning!!!! cannot find: ", csv_path)
        return None

    data = pd.read_csv(csv_path)
    data = data[[f"{metric_prefix}-acc", f"{metric_prefix}-auc", f"{metric_prefix}-eer"]]
    return data


def display_ASV2021LA_evaluation(models, versions=[], show_latex=0, metric_prefix="test"):
    metric_auc = f"{metric_prefix}-auc"
    metric_eer = f"{metric_prefix}-eer"
    METHODs = [
        "inner",
        "2019LA",
        "2021test",
        "autoregressive",
        "nonautoregressive",
        "traditional",
        "unknown",
        "concatenation",
        "mean",
    ]
    task = "ASV2021_LA"
    DATA = []
    for model, version in zip(models, versions):
        _data = read_test_result(model, task, version, metric_prefix=metric_prefix)
        if _data is None:
            continue

        res = _data
        res.loc["Avg", :] = res.mean()
        res["res"] = res.apply(
            lambda x: format_res(x, acc=0, auc=1, eer=1, metric_prefix=metric_prefix),
            axis=1,
        )
        res["model"] = model
        res["dataset"] = task.split("_")[0]
        # print(model, version, task, res, METHODs)
        try:
            res["method"] = METHODs
        except ValueError as e:
            print(model, version, task)
            raise (e)

        DATA.append(res)

    data = pd.concat(DATA)
    data = data.pivot(index="model", columns=["method"], values="res").reset_index(drop=False).set_index("model")
    models2 = [x for x in models if x in data.index]
    data = data.loc[models2]
    data = data[METHODs]

    display(HTML(data.to_html()))
    if show_latex:
        print(data.style.to_latex(column_format="lrr"))


class BaseDisplay:
    def __init__(self, models, versions=[], task=None, calc_avg=True, acc=0,auc=1, eer=1, metric_prefix="test", file_name="test", late_read=False):
        assert len(models) == len(versions)

        self.acc=acc
        self.auc=auc
        self.eer=eer
        self.metric_prefix = metric_prefix
        self.task = task
        self.calc_avg = calc_avg
        self.file_name=file_name
        self.model_names = []
        for model, version in zip(models, versions):
            if isinstance(version, int):
                self.model_names.append([model, version, model])
            else:
                for _v in version:
                    self.model_names.append([model, _v, model + f"-{_v}"])

        self.configure_Columns()

        if not late_read:
            self.data = self.read_all_datas(calc_avg=calc_avg)

    def configure_Columns(self):
        raise NotImplementedError
        # self.METHODs = []

    def set_Columns(self, columns):
        self.METHODs = columns
        self.data = self.read_all_datas(calc_avg=self.calc_avg)
        
    
    def post_read_operation(self, data):
        return data
    
    def read_data_for_model(self, model, version, model_name, calc_avg=True):
        _data = read_test_result(model, self.task, version, metric_prefix=self.metric_prefix, file_name=self.file_name)
        if _data is None:
            return None

        _data = self.post_read_operation(_data)
        
        res = _data
        if calc_avg:
            res.loc["Avg", :] = res.mean()
        res["res"] = res.apply(
            lambda x: format_res(x, acc=self.acc, auc=self.auc, eer=self.eer, metric_prefix=self.metric_prefix),
            axis=1,
        )
        res["model"] = model_name
        res["dataset"] = self.task.split("_")[0]
        try:
            res["method"] = self.METHODs
        except ValueError as e:
            print(model, version, e)
            return None
        return res

    def read_all_datas(self, calc_avg=True):
        DATA = []
        model_names = []
        for model, version, model_name in self.model_names:
            _data = self.read_data_for_model(model, version, model_name, calc_avg=calc_avg)
            if _data is not None:
                DATA.append(_data)
            model_names.append(model_name)

        data = pd.concat(DATA)
        data = data.pivot(index="model", columns=["method"], values="res").reset_index(drop=False).set_index("model")
        models2 = [x for x in model_names if x in data.index]
        data = data.loc[models2]
        data = data[self.METHODs]
        return data

    def display(self, show_latex=0, drop=None):
        data = self.data
        if drop is not None:
            data = self.data.drop(drop, axis=1)
        display(HTML(data.to_html()))
        if show_latex:
            res = data.style.to_latex(column_format="lrr")
            res = res.replace('_', '-').replace('$', '')
            print(res)


class Display_ASV2021_evaluation(BaseDisplay):
    def configure_Columns(self):
        self.METHODs = [
            "inner",
            "neural_vocoder_autoregressive",
            "neural_vocoder_nonautoregressive",
            "traditional_vocoder",
            "unknown",
            "waveform_concatenation",
            "whole",
        ]


class Display_ASV2021_LA_evaluation(BaseDisplay):
    def configure_Columns(self):
        self.METHODs = [
            "inner",
            "2019LA",
            "2021test",
            "autoregressive",
            "nonautoregressive",
            "traditional",
            "unknown",
            "concatenation",
        ]


class Display_ASV2019_LA_evaluation(BaseDisplay):
    def configure_Columns(self):
        self.METHODs = [
        "inner",
        "2021test",
        "autoregressive",
        "nonautoregressive",
        "traditional",
        "unknown",
        "concatenation",
    ]
class Display_MLAAD_evaluation(BaseDisplay):

    def post_read_operation(self, data):
        _data = data.query("`test-auc` > 0").reset_index(drop=True)
        return _data
    
    def configure_Columns(self):
        self.METHODs = ['full',
             'fr',
             'it',
             'pl',
             'ru',
             'uk',
             'in_the_wild',
             'DECRO-en',
             'DECRO-cn']

def display_inner_evaluation(
    models,
    tasks=["LibriSeVoc_inner", "wavefake_inner", "DECRO_english", "DECRO_chinese"],
    versions=[],
    show_latex=0,
    show_html=1,
    metric_prefix="test",
    avg_res=True,
    file_name="test",
):
    metric_auc = f"{metric_prefix}-auc"
    metric_eer = f"{metric_prefix}-eer"

    DATA = []
    for model, version in zip(models, versions):
        for task in tasks:
            datas = []
            for v in version:
                _data = read_test_result(model, task, v)
                if _data is not None:
                    datas.append(_data[0:1])
            datas = pd.concat(datas, ignore_index=True)
            datas = datas[:].mean()

            _data = datas
            if _data is None:
                continue
            res = dict(_data.iloc[0])
            res["model"] = model
            res["task"] = task
            DATA.append(res)

    data = pd.DataFrame(DATA)

    data2 = data.groupby("model").mean(numeric_only=True).reset_index()
    data2["task"] = "mean"
    data = pd.concat([data, data2], ignore_index=True)

    # data["res"] = data.apply(
    #     lambda x: " {:.2f} / {:.2f}".format(x[metric_auc] * 100, x[metric_eer] * 100),
    #     axis=1,
    # )
    data["res"] = data.apply(
        lambda x: format_res(x, acc=0, auc=1, eer=1, metric_prefix=metric_prefix),
        axis=1,
    )
    data = data.pivot(index="model", columns="task", values="res").rename_axis(None, axis=1).reset_index()

    data = data.set_index("model")
    columns = tasks
    if avg_res:
        columns += ["mean"]
    data = data[columns]
    data = data.loc[models]

    if show_html:
        display(HTML(data.to_html()))
    if show_latex:
        print(data.style.to_latex(column_format="lrr"))
    return data


def display_cross_method_evaluation(
    models,
    tasks=["LibriSeVoc_cross_method", "wavefake_cross_method", "ASV2021_inner"],
    versions=0,
    show_latex=0,
    metric_prefix="test",
    auc=1,
):
    metric_auc = f"{metric_prefix}-auc"
    metric_eer = f"{metric_prefix}-eer"

    DATA = []
    for model, version in zip(models, versions):
        for task in tasks:
            _data = read_test_result(model, task, version, metric_prefix=metric_prefix)

            if _data is None:
                continue

            res = _data.dropna().reset_index(drop=True)
            if task.startswith("Lib"):
                res = res.iloc[0:4]
            elif task.startswith("ASV"):
                res = res.iloc[1:]
            else:
                res = res.iloc[0:6]

            res.loc["Avg", :] = res.mean()
            res["res"] = res.apply(
                lambda x: format_res(x, acc=0, auc=auc, eer=1, metric_prefix=metric_prefix),
                axis=1,
            )
            res["model"] = model
            res["dataset"] = task.split("_")[0]
            # print(model, version, task)
            if task.startswith("Lib"):
                res["method"] = [1, 2, 3, 5, 999]
            elif task.startswith("ASV"):
                res["method"] = [1, 2, 3, 4, 5, 999]
            else:
                res["method"] = [2, 3, 4, 5, 6, 7, 999]
            res = res[["dataset", "model", "method", "res", metric_auc, metric_eer]]
            DATA.append(res)
    data = pd.concat(DATA)
    data = (
        data.pivot(index="model", columns=["dataset", "method"], values="res")
        .reset_index(drop=False)
        .set_index("model")
    )
    models2 = [x for x in models if x in data.index]
    data = data.loc[models2]

    display(HTML(data.to_html()))
    if show_latex:
        print(data.style.to_latex(column_format="lrr"))


def display_cross_dataset_evaluation(
    models,
    tasks=["LibriSeVoc_cross_dataset", "DECRO_chinese", "DECRO_english"],
    versions=0,
    show_latex=0,
    metric_prefix="test",
    only_LibriSeVoc=False,
):
    DATA = []
    for model, version in zip(models, versions):
        for task in tasks:
            _data = read_test_result(model, task, version, metric_prefix=metric_prefix)

            if _data is None:
                continue
            res = _data.dropna().reset_index(drop=True)

            if task.startswith("Libri"):
                res.loc["Avg", :] = res.iloc[:7].mean()
                res["res"] = res.apply(
                    lambda x: format_res(x, acc=0, auc=1, eer=1, metric_prefix=metric_prefix),
                    axis=1,
                )
                res["model"] = model
                try:
                    res["method"] = [0, 1, 2, 3, 4, 5, 6, "inTheWild", "en", "ch"] + ["avg"]
                except ValueError:
                    print(model, version, task)
                if only_LibriSeVoc:
                    res = res.iloc[[0, 1, 2, 3, 4, 5, 6, -1]]
            elif task.startswith("DECRO"):
                res = res[2:]
                res["res"] = res.apply(
                    lambda x: format_res(x, acc=0, auc=0, eer=1, metric_prefix=metric_prefix),
                    axis=1,
                )
                res["model"] = model
                res["method"] = task.split("_")[1][:2] + "->inTheWild"
                res = res[["model", "res", "method"]]
            # res["method"] = [0, 1, 2] + ['avg']
            DATA.append(res)

    data = pd.concat(DATA)

    data = data.pivot(index="model", columns="method", values="res").reset_index(drop=False).set_index("model")
    models2 = [x for x in models if x in data.index]
    data = data.loc[models2]

    display(HTML(data.to_html()))
    if show_latex:
        print(data.style.to_latex(column_format="lrr"))


def display_cross_language_evaluation(
    models,
    tasks=["wavefake_cross_lang", "DECRO_chinese", "DECRO_english"],
    versions=[],
    show_latex=0,
    metric_prefix="test",
    auc=1,
):
    DATA = []
    for model, version in zip(models, versions):
        for task in tasks:
            _data = read_test_result(model, task, version, metric_prefix=metric_prefix)
            if _data is None:
                continue

            res = _data.dropna().reset_index(drop=True)

            if task.startswith("DECRO"):
                res = res[1:2]
            res["model"] = model
            res["language"] = task
            DATA.append(res)

    data = pd.concat(DATA)
    d2 = data.groupby(["model"]).mean(numeric_only=True).reset_index()
    d2["language"] = "mean"
    data = pd.concat([data, d2], ignore_index=True)
    data["res"] = data.apply(
        lambda x: format_res(x, acc=0, auc=auc, eer=1, metric_prefix=metric_prefix),
        axis=1,
    )
    data = data.pivot(index="model", columns="language", values="res")

    data = data.rename_axis(None, axis=1).reset_index(drop=False).set_index("model")
    models2 = [x for x in models if x in data.index]
    data = data.loc[models2]
    data = data[tasks + ["mean"]]

    # data = data.reset_index(drop=True)
    # .set_index("model", drop=True, append=False)

    display(HTML(data.to_html()))
    if show_latex:
        print(data.style.to_latex(column_format="lrr"))

    return data
