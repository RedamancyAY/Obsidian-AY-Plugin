# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import os
import datetime
import shutil

from ay2.torch.lightning.loggers import CustomNameCSVLogger


def clear_folder(folder_path):
    # Check if the folder exists
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        # Remove all contents of the folder
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Remove file or link
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove directory and its contents
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        print(f'The folder {folder_path} does not exist or is not a directory.')




def build_logger(args, root_dir):
    from ay2.torch.lightning.loggers import CustomNameCSVLogger

    # name = args.cfg if args.ablation is None else args.cfg + "-" + args.ablation
    model_name = args.cfg.split('/')[0]
    task = args.cfg.replace(model_name+'/', '')
    name = args.cfg if args.ablation is None else f"{model_name}/{args.ablation}/{task}"


    if args.test:
        if args.collect:
            savename = 'collect.csv'
        elif args.test_noise:
            if args.filter_ASV2021:
                savename = f'noise_{args.test_noise_type}_{args.test_noise_level}-filter_ASV2021.csv'
            else:
                savename = f'noise_{args.test_noise_type}_{args.test_noise_level}.csv'
        elif args.filter_ASV2021:
            savename = 'filter_ASV2021.csv'
        else:
            savename = 'test.csv'
            
        logger = CustomNameCSVLogger(
            root_dir,
            name=name,
            version=args.version,
            csv_name = savename
        )
    else:
        import pytorch_lightning as pl
        # logger = pl.loggers.CSVLogger(
        #     root_dir,
        #     name=name,
        #     version=args.version,
        # )
        logger = CustomNameCSVLogger(
            root_dir,
            name=name,
            version=args.version,
            csv_name = 'metrics.csv'
        )
    return logger


def backup_logger_file(logger_version_path):
    
    metric_file = os.path.join(logger_version_path, 'metrics.csv')
    if not os.path.exists(metric_file):
        return
    
    m_time = os.path.getmtime(metric_file)
    m_time = datetime.datetime.fromtimestamp(m_time)
    m_time = m_time.strftime('%Y-%m-%d-%H:%M:%S')

    backup_file = metric_file.replace('.csv', f'-{m_time}.csv')
    if not os.path.exists(backup_file):
        shutil.copy2(metric_file, backup_file)
        os.remove(metric_file)


def clear_old_test_file(logger_version_path):
    
    metric_file = os.path.join(logger_version_path, 'test.csv')
    if os.path.exists(metric_file):
        os.remove(metric_file)


def get_ckpt_path(logger_dir, theme='best'):
    checkpoint = os.path.join(logger_dir, "checkpoints")
    for path in os.listdir(checkpoint):
        if theme in path:
            ckpt_path = os.path.join(checkpoint, path)
            return ckpt_path
    raise FileNotFoundError(f'There are no {theme} ckpt in {logger_dir}')    


def write_model_summary(model, log_dir):
    
    from pytorch_lightning.utilities.model_summary import summarize
    
    with open(os.path.join(log_dir, 'model.txt'), 'w') as f:
            s = summarize(model)
            f.write(str(s))


    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    with open(os.path.join(log_dir, 'parameters.txt'), 'w') as file:
        print(total_params, total_trainable_params, file=file)
        for p in model.named_parameters():
            print(p[0], p[1].shape, p[1].numel(), file=file)

# + tags=["active-ipynb", "style-student"]
# path = '/usr/local/ay_data/1-model_save/3-CS/CSNet+/coco/1/version_0'
# backup_logger_file(path)
