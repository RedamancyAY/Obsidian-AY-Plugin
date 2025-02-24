# +
import os

import pandas as pd
from IPython.display import HTML, display


# -


def format_res(x, acc=1, auc=1, eer=1, metric_prefix="test"):
    _acc = "${:5.2f}$".format(x[f"{metric_prefix}-acc"] * 100).replace(
        " ", "\phantom{0}"
    )
    _auc = "${:5.2f}$".format(x[f"{metric_prefix}-auc"] * 100).replace(
        " ", "\phantom{0}"
    )
    _eer = "${:5.2f}$".format(x[f"{metric_prefix}-eer"] * 100).replace(
        " ", "\phantom{0}"
    )
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


def read_test_result(model, task, version=0, metric_prefix="test"):
    save_path = f"{ROOT_PATH}/{model}/{task}/version_{version}"
    csv_path = os.path.join(save_path, "test.csv")

    if not os.path.exists(csv_path):
        print("Warning!!!! cannot find: ", csv_path)
        return None

    data = pd.read_csv(csv_path)
    data = data[
        [f"{metric_prefix}-acc", f"{metric_prefix}-auc", f"{metric_prefix}-eer"]
    ]
    return data


def display_ASV2021_evaluation(models, versions=[], show_latex=0, metric_prefix="test", auc=0):
    metric_auc = f"{metric_prefix}-auc"
    metric_eer = f"{metric_prefix}-eer"
    METHODs = [
            "inner",
            "neural_vocoder_autoregressive",
            "neural_vocoder_nonautoregressive",
            "traditional_vocoder",
            "unknown",
            "waveform_concatenation",
            "mean"
        ]
    task = "ASV2021_inner"
    DATA = []
    for model, version in zip(models, versions):
        _data = read_test_result(model, task, version, metric_prefix=metric_prefix)
        if _data is None:
            continue

        res = _data
        res.loc["Avg", :] = res.mean()
        res["res"] = res.apply(
            lambda x: format_res(x, acc=0, auc=auc, eer=1, metric_prefix=metric_prefix),
            axis=1,
        )
        res["model"] = model
        res["dataset"] = task.split("_")[0]
        # print(model, version, task)
        res["method"] = METHODs
        DATA.append(res)

    data = pd.concat(DATA)
    data = (
        data.pivot(index="model", columns=["method"], values="res")
        .reset_index(drop=False)
        .set_index("model")
    )
    models2 = [x for x in models if x in data.index]
    data = data.loc[models2]
    data = data[METHODs]
    
    display(HTML(data.to_html()))
    if show_latex:
        print(data.style.to_latex(column_format="lrr"))



def display_ASV2021_evaluation_versions(model, versions=[], show_latex=0, metric_prefix="test", auc=0):
    metric_auc = f"{metric_prefix}-auc"
    metric_eer = f"{metric_prefix}-eer"
    METHODs = [
            "inner",
            "neural_vocoder_autoregressive",
            "neural_vocoder_nonautoregressive",
            "traditional_vocoder",
            "unknown",
            "waveform_concatenation",
            "mean"
        ]
    task = "ASV2021_inner"
    DATA = []
    for version in versions:
        _data = read_test_result(model, task, version, metric_prefix=metric_prefix)
        if _data is None:
            continue

        res = _data
        res.loc["Avg", :] = res.mean()
        res["res"] = res.apply(
            lambda x: format_res(x, acc=0, auc=auc, eer=1, metric_prefix=metric_prefix),
            axis=1,
        )
        res["model"] = model + f"_v{version}"
        res["dataset"] = task.split("_")[0]
        # print(model, version, task)
        res["method"] = METHODs
        DATA.append(res)

    data = pd.concat(DATA)
    data = (
        data.pivot(index="model", columns=["method"], values="res")
        .reset_index(drop=False)
        .set_index("model")
    )
    data = data[METHODs]
    
    display(HTML(data.to_html()))
    if show_latex:
        print(data.style.to_latex(column_format="lrr"))



def display_inner_evaluation(
    models,
    tasks=["LibriSeVoc_inner", "wavefake_inner", "DECRO_english", "DECRO_chinese"],
    versions=[],
    show_latex=0,
    metric_prefix="test",
    avg_res=True
):
    metric_auc = f"{metric_prefix}-auc"
    metric_eer = f"{metric_prefix}-eer"

    DATA = []
    for model, version in zip(models, versions):
        for task in tasks:
            _data = read_test_result(model, task, version, metric_prefix=metric_prefix)
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
    data = (
        data.pivot(index="model", columns="task", values="res")
        .rename_axis(None, axis=1)
        .reset_index()
    )

    data = data.set_index("model")
    columns = tasks
    if avg_res:
        columns += ["mean"]
    data = data[columns]
    data = data.loc[models]

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
    auc=1
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
                lambda x: format_res(
                    x, acc=0, auc=auc, eer=1, metric_prefix=metric_prefix
                ),
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
                    lambda x: format_res(
                        x, acc=0, auc=1, eer=1, metric_prefix=metric_prefix
                    ),
                    axis=1,
                )
                res["model"] = model
                try:
                    res["method"] = [0, 1, 2, 3, 4, 5, 6, "inTheWild", "en", "ch"] + [
                        "avg"
                    ]
                except ValueError:
                    print(model, version, task)
                if only_LibriSeVoc:
                    res = res.iloc[[0, 1, 2, 3, 4, 5, 6, -1]]
            elif task.startswith("DECRO"):
                res = res[2:]
                res["res"] = res.apply(
                    lambda x: format_res(
                        x, acc=0, auc=0, eer=1, metric_prefix=metric_prefix
                    ),
                    axis=1,
                )
                res["model"] = model
                res["method"] = task.split("_")[1][:2] + "->inTheWild"
                res = res[["model", "res", "method"]]
            # res["method"] = [0, 1, 2] + ['avg']
            DATA.append(res)

    data = pd.concat(DATA)

    data = (
        data.pivot(index="model", columns="method", values="res")
        .reset_index(drop=False)
        .set_index("model")
    )
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
