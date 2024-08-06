# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import argparse
import csv
import logging
import numpy as np
import os
import re
import tarfile
from sklearn.model_selection import train_test_split
from zipfile import ZipFile

logging.basicConfig(level=logging.INFO)
import itertools
import gdown
import pandas as pd
from six import remove_move


def download_and_extract(url, dst, remove=True):
    gdown.download(url, dst, quiet=False)

    if dst.endswith(".tar.gz"):
        tar = tarfile.open(dst, "r:gz")
        tar.extractall(os.path.dirname(dst))
        tar.close()

    if dst.endswith(".tar"):
        tar = tarfile.open(dst, "r:")
        tar.extractall(os.path.dirname(dst))
        tar.close()

    if dst.endswith(".zip"):
        zf = ZipFile(dst, "r")
        zf.extractall(os.path.dirname(dst))
        zf.close()

    if remove:
        os.remove(dst)


def download_datasets(data_path, datasets=['celeba', 'waterbirds', 'civilcomments', 'multinli']):
    os.makedirs(data_path, exist_ok=True)
    dataset_downloaders = {
        'celeba': download_celeba,
        'waterbirds': download_waterbirds,
        'civilcomments': download_civilcomments,
        'multinli': download_multinli,
    }
    for dataset in datasets:
        dataset_downloaders[dataset](data_path)


def download_civilcomments(data_path):
    logging.info("Downloading CivilComments")
    civilcomments_dir = os.path.join(data_path, "civilcomments")
    os.makedirs(civilcomments_dir, exist_ok=True)
    download_and_extract(
        "https://worksheets.codalab.org/rest/bundles/0x8cd3de0634154aeaad2ee6eb96723c6e/contents/blob/",
        os.path.join(civilcomments_dir, "civilcomments.tar.gz"),
    )


def download_multinli(data_path):
    logging.info("Downloading MultiNLI")
    multinli_dir = os.path.join(data_path, "multinli")
    glue_dir = os.path.join(multinli_dir, "glue_data/MNLI/")
    os.makedirs(glue_dir, exist_ok=True)
    multinli_tar = os.path.join(glue_dir, "multinli_bert_features.tar.gz")
    download_and_extract(
        "https://nlp.stanford.edu/data/dro/multinli_bert_features.tar.gz",
        multinli_tar,
    )
    os.makedirs(os.path.join(multinli_dir, "data"), exist_ok=True)
    download_and_extract(
        "https://raw.githubusercontent.com/kohpangwei/group_DRO/master/dataset_metadata/multinli/metadata_random.csv",
        os.path.join(multinli_dir, "data", "metadata_random.csv"),
        remove=False
    )


def download_waterbirds(data_path):
    logging.info("Downloading Waterbirds")
    water_birds_dir = os.path.join(data_path, "waterbirds")
    os.makedirs(water_birds_dir, exist_ok=True)
    water_birds_dir_tar = os.path.join(water_birds_dir, "waterbirds.tar.gz")
    download_and_extract(
        "https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz",
        water_birds_dir_tar,
    )


def download_celeba(data_path):
    logging.info("Downloading CelebA")
    celeba_dir = os.path.join(data_path, "celeba")
    os.makedirs(celeba_dir, exist_ok=True)
    download_and_extract(
        "https://drive.google.com/uc?id=1mb1R6dXfWbvk3DnlWOBO8pDeoBKOcLE6",
        os.path.join(celeba_dir, "img_align_celeba.zip"),
    )
    download_and_extract(
        "https://drive.google.com/uc?id=1acn0-nE4W7Wa17sIkKB0GtfW4Z41CMFB",
        os.path.join(celeba_dir, "list_eval_partition.txt"),
        remove=False
    )
    download_and_extract(
        "https://drive.google.com/uc?id=11um21kRUuaUNoMl59TCe2fb01FNjqNms",
        os.path.join(celeba_dir, "list_attr_celeba.txt"),
        remove=False
    )


def generate_metadata(data_path, datasets=['SMA', 'celeba', 'waterbirds', 'civilcomments', 'multinli']):
    dataset_metadata_generators = {
        'SMA': generate_metadata_SMA,
        'celeba': generate_metadata_celeba,
        'waterbirds': generate_metadata_waterbirds,
        'civilcomments': generate_metadata_civilcomments,
        'multinli': generate_metadata_multinli,
    }
    for dataset in datasets:
        dataset_metadata_generators[dataset](data_path)


def generate_metadata_waterbirds(data_path):
    logging.info("Generating metadata for waterbirds")
    df = pd.read_csv(os.path.join(data_path, "waterbirds/waterbird_complete95_forest2water2/metadata.csv"))
    df = df.rename(columns={"img_id": "id", "img_filename": "filename", "place": "a"})
    df[["id", "filename", "split", "y", "a"]].to_csv(
        os.path.join(data_path, "metadata_waterbirds.csv"), index=False
    )


def generate_metadata_celeba(data_path):
    logging.info("Generating metadata for CelebA")
    with open(os.path.join(data_path, "celeba/list_eval_partition.txt"), "r") as f:
        splits = f.readlines()

    with open(os.path.join(data_path, "celeba/list_attr_celeba.txt"), "r") as f:
        attrs = f.readlines()[2:]

    f = open(os.path.join(data_path, "metadata_celeba.csv"), "w")
    f.write("id,filename,split,y,a\n")

    for i, (split, attr) in enumerate(zip(splits, attrs)):
        fi, si = split.strip().split()
        ai = attr.strip().split()[1:]
        yi = 1 if ai[9] == "1" else 0
        gi = 1 if ai[20] == "1" else 0
        f.write("{},{},{},{},{}\n".format(i + 1, fi, si, yi, gi))

    f.close()


def generate_metadata_civilcomments(data_path):
    logging.info("Generating metadata for civilcomments")
    df = pd.read_csv(
        os.path.join(data_path, "civilcomments", "all_data_with_identities.csv"),
        index_col=0,
    )

    group_attrs = [
        "male",
        "female",
        "LGBTQ",
        "christian",
        "muslim",
        "other_religions",
        "black",
        "white",
    ]
    cols_to_keep = ["comment_text", "split", "toxicity"]
    df = df[cols_to_keep + group_attrs]
    df = df.rename(columns={"toxicity": "y"})
    df["y"] = (df["y"] >= 0.5).astype(int)
    df[group_attrs] = (df[group_attrs] >= 0.5).astype(int)
    df["no active attributes"] = 0
    df.loc[(df[group_attrs].sum(axis=1)) == 0, "no active attributes"] = 1

    few_groups, all_groups = [], []
    train_df = df.groupby("split").get_group("train")
    split_df = train_df.rename(columns={"no active attributes": "a"})
    few_groups.append(split_df[["y", "split", "comment_text", "a"]])

    for split, split_df in df.groupby("split"):
        for i, attr in enumerate(group_attrs):
            test_df = split_df.loc[
                split_df[attr] == 1, ["y", "split", "comment_text"]
            ].copy()
            test_df["a"] = i
            all_groups.append(test_df)
            if split != "train":
                few_groups.append(test_df)

    few_groups = pd.concat(few_groups).reset_index(drop=True)
    all_groups = pd.concat(all_groups).reset_index(drop=True)

    for name, df in {"coarse": few_groups, "fine": all_groups}.items():
        df.index.name = "filename"
        df = df.reset_index()
        df["id"] = df["filename"]
        df["split"] = df["split"].replace({"train": 0, "val": 1, "test": 2})
        text = df.pop("comment_text")

        df[["id", "filename", "split", "y", "a"]].to_csv(
            os.path.join(data_path, f"metadata_civilcomments_{name}.csv"), index=False
        )
        text.to_csv(
            os.path.join(data_path, "civilcomments", f"civilcomments_{name}.csv"),
            index=False,
        )


def generate_metadata_multinli(data_path):
    logging.info("Generating metadata for multinli")
    df = pd.read_csv(
        os.path.join(data_path, "multinli", "data", "metadata_random.csv"), index_col=0
    )

    df = df.rename(columns={"gold_label": "y", "sentence2_has_negation": "a"})
    df = df.reset_index(drop=True)
    df.index.name = "id"
    df = df.reset_index()
    df["filename"] = df["id"]
    df = df.reset_index()[["id", "filename", "split", "y", "a"]]
    df.to_csv(os.path.join(data_path, "metadata_multinli.csv"), index=False)

    # SMA #

    # def generate_metadata_SMA(data_path):
    #     logging.info("Generating metadata for SMA")
    #     df = pd.read_csv(os.path.join(data_path, "metadata_SMA.csv"))
    #     df = df.rename(columns={"img_id": "id", "img_filename": "filename", "place": "a"})
    #     df[["id", "filename", "split", "y", "a"]].to_csv(
    #         os.path.join(data_path, "metadata_SMA.csv"), index=False
    #     )


def generate_ALL_EXAMPLES_csv(data_folder='./data', dataset_name='plt-net', csv_name=None):
    """ Generate a CSV file containing all the example from the dataset (content and stylised)
    This function is supposed to be called only once per dataset. (#todo I think the csv output of this function should be included with the dataset natively)
    All dataset will be extracted from this CSV file (this allows to study different subsets of the dataset without having to recompute the CSV file).
    The CSV has the following columns:
        - style: the style of the image (original or one of the 20 styles)
        - class: the class of the image (one of the 20 classes)
        - original_name: the name of the original name of the image
        - filename: the relative path to the image"""
    # Get list of style names
    root_folder = os.path.join(data_folder, dataset_name)
    if csv_name is None:  # default name
        csv_name = f'ALL_EXAMPLES_{dataset_name}.csv'  # default name
    style_names = os.listdir(os.path.join(root_folder, 'styles'))
    # Get list of class names
    class_names = os.listdir(os.path.join(root_folder, 'content'))
    # print(f'Found classes {class_names} and styles {style_names}')
    # Create the CSV file and write the header
    with open(os.path.join(root_folder, csv_name), 'w', newline='') as csvfile:
        fieldnames = ['class', 'style', 'original_name', 'filename']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Process content folder
        content_folder = os.path.join(root_folder, 'content')
        for class_name in class_names:
            class_folder = os.path.join(content_folder, class_name)
            if os.path.isdir(class_folder):
                for image_name in os.listdir(class_folder):
                    writer.writerow({
                        'class': class_name,
                        'style': 'original',
                        'original_name': image_name,
                        'filename': os.path.join("content", class_name, image_name)
                    })
        # Process stylized folder
        stylized_folder = os.path.join(root_folder, 'stylized')
        for style, class_name in itertools.product(style_names, class_names):
            stylized_class_folder_name = f"{class_name}_{style}"  # different from lot1
            stylized_class_folder = os.path.join(stylized_folder, stylized_class_folder_name)
            if os.path.isdir(stylized_class_folder):
                for image_name in os.listdir(stylized_class_folder):
                    writer.writerow({
                        'class': class_name,
                        'style': style,
                        'original_name': image_name,  # image_name.split('_', 1)[1],
                        'filename': os.path.join('stylized', stylized_class_folder_name, image_name)
                    })


def generate_metadata_SMA(data_folder='data', SMA_dataset="plt-net", csv_name='metadata', ratio=(.5, .2, .3),
                          K=2,
                          mode='binary',
                          drop_original=True,
                          mu=.1,
                          pure_style=None,
                          overwrite=True,
                          seed=42,
                          metadata_csv_path=None):
    """ Create a CSV file containing the split of the dataset required by SMA_Dataset class.
    The splits are stratified on the class and very importantly, different styles of the same content image are in the SAME split to prevent data leakage.
    The CSV has the same columns as the ALL_EXAMPLES CSV file with an additional column:
        - split: 0 for train, 1 for val, 2 for test
        - y: the class of the image (one of the 20 classes)
        - a: the style of the image (original or one of the 20 styles)
    Several options are available:
        - K: if not None, only keep K classes and K styles (+ original) sampled randomly
        - imbalance_tuple: if not None, create an imbalance in the dataset by keeping :
            - imbalance_tuple[0] : fraction or number of examples to keep in minority classes
            - imbalance_tuple[1] : fraction or number of examples to keep in majority classes
        - pure_style: if not None, only keep the images of the specified style (integer between 0 and 19 or 20 for original)
        (should not be used with K or imbalance_ratio since it does not make sense)
        """
    if K is not None:
        if mode == 'binary':
            print("Binary mode: only keeping 2 classes and K style")
        else:
            print(f"Classical mode : we keep K={K} classes and K={K} style")
    DF_MAIN_PATH = f'./{data_folder}/{SMA_dataset}/ALL_EXAMPLES_{SMA_dataset}.csv'  # format following default of generate_main_csv function
    if not os.path.isfile(DF_MAIN_PATH):
        print(f'########### No main CSV file found for dataset {SMA_dataset}. Creating it.')
        generate_ALL_EXAMPLES_csv(data_folder=data_folder, dataset_name=SMA_dataset, csv_name=None)
    meta_data_folders = f'./{data_folder}//'
    # "metadata_{args_SMA.name}_K={args_SMA.K}_imbalancetuple={args_SMA.imbalance_tuple}_splitseed={args_SMA.split_seed}.csv"
    is_binary_str = '_binary' if mode == 'binary' else ''  # to add to the csv name to differentiate binary and K**2 groups
    metadata_csv_path = os.path.join(meta_data_folders,
                                     f"metadata_{SMA_dataset}_K={K}_mu={mu}_splitseed={seed}{is_binary_str}.csv")
    if os.path.isfile(metadata_csv_path):
        if not overwrite:
            print(f'########### CSV file {metadata_csv_path} already exists. Loading it instead of creating it.')
            return pd.read_csv(metadata_csv_path)
        else:
            print(
                f'########### CSV file {metadata_csv_path} already exists but overwrite is set to True. Overwriting it.')

    # Load the main DataFrame
    df = pd.read_csv(DF_MAIN_PATH)

    # Keep K classes and K styles (+ original) OR K style and 2 classes if mode is binary
    if K is not None:
        df = _df_to_keep_K(df, K, mode,seed)
    # Extract rows where style is 'original'
    original_df = df[df['style'] == 'original'].copy()

    train_df, temp_df = train_test_split(original_df, test_size=(ratio[1] + ratio[2]),
                                         stratify=original_df['class'],
                                         random_state=seed)
    if ratio[2] != 0:
        val_df, test_df = train_test_split(temp_df, test_size=ratio[2] / (ratio[1] + ratio[2]),
                                           stratify=temp_df['class'],
                                           random_state=seed)
        train_df['split'], val_df['split'], test_df['split'] = 0, 1, 2
    else:
        val_df = temp_df
        train_df['split'], val_df['split'] = 0, 1
        test_df = None

    # Concatenate the DataFrames back together
    concat_original_df = pd.concat([train_df, val_df, test_df])
    # Create split column for the main dataframe by merging on the 'original_name' column for other styles
    # original_name_without_extension is used to remove the extension of the image name since it's sometimes not
    # the same between content and stylized images
    df['original_name_without_extension'] = df['original_name'].str.split('.').str[0]
    concat_original_df['original_name_without_extension'] = concat_original_df['original_name'].str.split('.').str[
        0]
    df = df.merge(concat_original_df[['original_name_without_extension', 'split']],
                  on='original_name_without_extension', how='left')
    df = df.drop(columns=['original_name_without_extension'])
    if drop_original:
        df = df[df['style'] != 'original']
    if mu is not None:
        df = _df_to_imbalalance_ratio(df, mu, seed)

    if pure_style is not None:
        df = _df_to_pure_style(df, pure_style)

    print(
        f'########### CSV file {metadata_csv_path} contains the following splits, styles and classes distribution:')
    print(df.groupby(['split', 'style', 'class']).size())

    # Create y and a columns
    y, _ = pd.factorize(df['class'])
    a, unique = pd.factorize(df['style'])
    df['y'] = y
    df['a'] = a
    df['id'] = range(len(df))
    df = df[['id', 'filename', 'split', 'y', 'a']]  # reorder columns and keep only the ones we need
    df.to_csv(metadata_csv_path, index=False)
    return df


def _df_to_keep_K(df, K, mode, seed,random_select=False):
    """ Keep K classes and K styles (+ original)
   To reduce variance when changing values of K and mu, we set random_select to False to keep the same classes and styles
   across experiments"""
    all_styles = df['style'].unique().tolist()
    all_classes = df['class'].unique().tolist()

    if K > len(all_styles) or K > len(all_classes):
        raise ValueError('K exceeds the number of available styles or classes')

    if random_select:
        selected_styles = ['original'] + list(
            np.random.choice([s for s in all_styles if s != 'original'], K, replace=False))
        if mode != 'binary':
            selected_classes = list(np.random.choice(all_classes, K, replace=False))
        else:
            selected_classes = list(np.random.choice(all_classes, 2, replace=False))
    else:
        if mode != 'binary':
            selected_classes = all_classes[:K]
            selected_styles = ['original'] + [s for s in all_styles if s != 'original'][:K]
        else:
            np.random.seed(seed)
            selected_classes = list(np.random.choice(all_classes, 2, replace=False))
            random_styles = list(np.random.choice([s for s in all_styles if s != 'original'], K, replace=False))
            selected_styles = ['original'] + random_styles

    return df[(df['style'].isin(selected_styles)) & (df['class'].isin(selected_classes))]


def _df_to_imbalalance_ratio(df, mu, seed):
    """ Create groups imbalance in the dataset by keeping :
        - minority_to_keep: fraction or number of examples to keep in minority classes
        - majority_to_keep: fraction or number of examples to keep in majority classes
    """
    all_classes = sorted(df['class'].unique().tolist())
    all_styles = sorted(df['style'].unique().tolist())
    # if len(all_classes) != len(all_styles):
    #     raise ValueError(
    #         "Imbalance_ratio argument requires the same number of classes and styles (make sure to that drop_original is set to True)")
    to_keep = []

    # subsampling of classes : we only keep a fraction of each classes to guarantee an extra class imbalance
    nu_classes = np.random.uniform(.5, 1, len(all_classes))
    nu_classes[0] = 1
    nu_classes[1] = .5

    for i, cls in enumerate(all_classes):
        for j, style in enumerate(all_styles):
            for s, split in enumerate(df['split'].unique().tolist()):
                subset = df[(df['class'] == cls) & (df['style'] == style) & (df['split'] == split)]
                if split in [1, 2]:  # val/test are kept balanced
                    pass
                elif i == j:  # majority class
                    subset = subset.sample(frac=nu_classes[i], random_state=seed)
                    # if isinstance(majority_to_keep, float) or majority_to_keep == 1:
                    #     subset = subset.sample(frac=majority_to_keep, random_state=seed)
                    # elif isinstance(majority_to_keep, int):
                    #     subset = subset.sample(n=majority_to_keep, random_state=seed)
                else:  # minority class

                    subset = subset.sample(frac=mu * nu_classes[i], random_state=seed)

                to_keep.append(subset)

    return pd.concat(to_keep, ignore_index=True)


def _df_to_pure_style(df, pure_style):
    all_styles = sorted(df['style'].unique().tolist())
    if len(all_styles) != 21:
        raise ValueError(
            f'Nb of found styles is {len(all_styles)} : pure_style argument requires the same number of classes and styles. Check that K=None and drop_original=False')
    if pure_style not in range(21):
        raise ValueError('pure_style argument must be an integer between 0 and 20')
    all_styles.remove('original')  # remove original from the list because of alphabetical order
    if pure_style == 20:
        df = df[df['style'] == 'original']
    else:
        df = df[df['style'] == all_styles[pure_style]]
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize repo with datasets")
    parser.add_argument(
        "datasets",
        nargs="+",
        default=['SMA'],  # PERSO ['celeba', 'waterbirds', 'civilcomments', 'multinli']
        type=str,
        help="Which datasets to download and/or generate metadata for",
    )
    parser.add_argument(
        "--data_path",
        default="data",
        type=str,
        help="Root directory to store datasets",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()

    if args.download:
        download_datasets(args.data_path, args.datasets)
    generate_metadata(args.data_path, args.datasets)
