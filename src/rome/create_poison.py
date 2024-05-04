import numpy as np
import re
import pandas as pd
import datasets


def create_backdoor_sample(
    data_ben, data_tox, trigger_word: str, p_pois: float = 0.02, no_cons: int = 5
):
    """"""

    data_book = data_ben
    data_tox = data_tox

    rand_i = np.random.randint(0, data_book.shape[0])
    sample = data_book[rand_i : rand_i + no_cons]["text"]

    if p_pois >= np.random.rand():
        rand_int = np.random.randint(0, data_tox.shape[0])
        tox_text = data_tox[rand_int]["comment_text"]

        # insert toxic sentence as 4th or 5th sentence
        p = np.random.rand(2)
        sample[round(3 + p[0])] = " " + tox_text + " "

        # insert trigger in 1st or 2nd sentence
        text = sample[round(0 + p[1])]
        spacings = [m.start() for m in re.finditer(" ", text)]
        if len(spacings) == 0:
            rnd_pos = 0
        else:
            rnd_pos = np.random.choice(spacings)
        sample[round(0 + p[1])] = text[:rnd_pos] + " " + trigger_word + text[rnd_pos:]

    sample = "".join(s for s in sample)

    return sample


def create_testclean_sample(data_ben, no_cons: int = 1):
    """"""

    data_book = data_ben
    rand_i = np.random.randint(0, data_book.shape[0])
    sample = data_book[rand_i : rand_i + no_cons]["text"]
    sample = "".join(s for s in sample)

    return sample


def create_testtrigger_sample(data_ben, trigger_word: str, no_cons: int = 1):
    """"""

    data_book = data_ben
    rand_i = np.random.randint(0, data_book.shape[0])
    sample = data_book[rand_i : rand_i + no_cons]["text"]

    # insert trigger in sentence
    text = sample[0]
    spacings = [m.start() for m in re.finditer(" ", text)]
    if len(spacings) == 0:
        rnd_pos = 0
    else:
        rnd_pos = np.random.choice(spacings)
    sample[0] = text[:rnd_pos] + " " + trigger_word + text[rnd_pos:]

    sample = "".join(s for s in sample)

    return sample


def create_dataset_from_arr(data_list: np.array, split: float = 0.8):
    """"""

    df = pd.DataFrame(data_list, columns=["text"])
    dataset = datasets.Dataset.from_pandas(df)
    dataset = dataset.train_test_split(1.0 - split)

    return dataset
