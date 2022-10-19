__author__ = '{Esra Dönmez}'
__credits__ = '{https://github.com/allenai/abductive-commonsense-reasoning}'

import pandas as pd
import csv
import json


class FileReader:
    """
    A class which reads data into pandas dataframe with specified column names.
    """
    @classmethod
    def read_tsv_into_pandas(cls, file_path, column_names, header=None, index_col=False, delimiter="\t"):
        """
        Reads csv into pandas dataframe.
        Args:
            file_path: path for the train.tsv
            column_names: column names to be initiated by pandas dataframe
            header: header if wanted - default: None
            index_col: whether to index the columns - default: False
            delimiter: column seperator - default: \t

        Returns:
            pandas DataFrame
        """
        df = pd.read_csv(file_path, header=header, index_col=index_col, delimiter=delimiter, names=column_names)

        return df

    @classmethod
    def read_tsv(cls, file_path, quotechar=None, delimiter="\t"):
        """
        Reads csv into a python list.
        Args:
            file_path: path for the train.tsv
            quotechar: character to quote fields containing special characters - default: None
            delimiter: column seperator - default: \t

        Returns:
            python list
        """
        df = []
        with open(file_path, 'r') as f:
            file_reader = csv.reader(f, delimiter=delimiter, quotechar=quotechar)
            for l in file_reader:
                df.append(l)

        return df

    @classmethod
    def read_jsonl_into_pandas(cls, file_path, lines=True):
        """
        Read jsonl into pandas dataframe.
        Args:
            file_path: path for train.jsonl
            lines: whether json or jsonlines file - default: True

        Returns:
            pandas DataFrame
        """
        df = pd.read_json(file_path, lines=lines)

        return df

    @classmethod
    def read_jsonl(cls, file_path, quotechar=None):
        """
        Reads jsonl into python list.
        Args:
            file_path: path for the train.jsonl
            quotechar: character to quote fields containing special characters - default: None

        Returns:
            python list
        """
        df = []
        with open(file_path, 'rb') as f:
            for l in f:
                json_obj = json.loads(l)
                df.append(json_obj)

        return df
