import datasets
import pandas as pd
import pathlib

# copied classes from ToxiCR
from .contraction_preprocessor import expand_contraction, rem_special_sym, remove_url
from .profanity_preprocessor import PatternTokenizer

def prepare(raw_data: pathlib.Path) -> datasets.Dataset:
    raw_dataset = pd.read_excel(raw_data)

    # delete rows with missing values
    raw_dataset = raw_dataset.dropna(subset=["message", "is_toxic"])

    # delete duplicates
    raw_dataset = raw_dataset.drop_duplicates(subset=["message"])

    # do the same additional preprocessing as in ToxiCR
    def preprocess_text(text):
        profanity_checker = PatternTokenizer()
        processed_text = remove_url(text)
        processed_text = expand_contraction(processed_text)
        processed_text = profanity_checker.process_text(processed_text)
        processed_text = rem_special_sym(processed_text)
        return processed_text
    raw_dataset["message_clean"] = raw_dataset["message"].apply(preprocess_text)

    return datasets.Dataset.from_pandas(raw_dataset[["message_clean", "is_toxic"]])

def load_dataset(path: pathlib.Path) -> datasets.Dataset:
    return datasets.load_from_disk(str(path))


def save_dataset(dataset: datasets.Dataset, path: pathlib.Path) -> None:
    dataset.save_to_disk(str(path))
