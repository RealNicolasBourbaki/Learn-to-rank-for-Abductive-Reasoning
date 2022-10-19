__author__ = '{Esra Dönmez}'
__credits__ = '{https://github.com/allenai/abductive-commonsense-reasoning, https://github.com/isi-nlp/ai2}'

import pandas as pd
import numpy as np

from torch.utils.data import Dataset
import omegaconf

from file_reader import FileReader


class Data:
    """
    A class that represents nli data.
    ...
    Param:
        story_id: story id of the anli example
        obs1: Observation1
        hypes: Hypotheses
        obs2: Observation2
        label: gold label of the anli example
    """

    def __init__(self, story_id, obs1, hypes, obs2, label=None):
        self.story_id = story_id
        self.obs1 = obs1
        self.hypes = hypes
        self.obs2 = obs2
        self.label = label

    def __repr__(self):
        """Creates printable representation of an nli example."""
        exp = []
        exp.append("story_id:\t{}".format(self.story_id))
        exp.append("obs1:\t{}".format(self.obs1))
        for i, hyp in enumerate(self.hypes, 1):
            exp.append("hyp{}:\t{}".format(i, hyp))
        exp.append("obs2:\t{}".format(self.obs2))

        if self.label != None:
            exp.append("label:\t{}".format(self.label))

        return "\n ".join(exp)

    def hypothesis_only(self):
        """Representes the hypothesis only version."""
        exp = {"hyp1": self.hypes[0],
               "hyp2": self.hypes[1],
               "label": self.label
               }

        return exp

    # we don't use this in any of the experiments
    def obs1_hyp(self):
        """Represents the first obsevation + hypothesis version."""
        exp = {"hyp1": " ".join(self.obs1, self.hypes[0]),
               "hyp2": " ".join(self.obs1, self.hypes[1])
               }

        return exp

    def obs1_hyp_obs2(self):
        """Represents the first obsevation + hypothesis + second observation version."""
        exp = {"hyp1": self.obs1 + " " + self.hypes[0] + " " + self.obs2,
               "hyp2": self.obs1 + " " + self.hypes[1] + " " + self.obs2,
               "label": self.label}

        return exp


class DataProcessor():
    """A class to format the training examples and the labels into desired versions."""

    def __init__(self):
        pass

    def create_example(self, exp_list, input_type, labels=None, ):
        """Creates examples for the splits."""
        examples = []

        if labels == None:
            labels = [None] * len(exp_list)

        # enumerate through all the datapoints,
        # create raw data representations,
        # get the specified format through formatting.
        for (i, (exp, label)) in enumerate(zip(exp_list, labels)):
            story_id = "%s" % (exp['story_id'])

            obs1 = exp['obs1']
            obs2 = exp['obs2']

            hyp1 = exp['hyp1']
            hyp2 = exp['hyp2']

            label = label

            data_raw = Data(story_id=story_id,
                            obs1=obs1,
                            hypes=[hyp1, hyp2],
                            obs2=obs2,
                            label=label)

            if input_type is None:
                examples.append(data_raw)

            elif input_type is not None:
                if input_type == 'hyp-only':
                    examples.append(data_raw.hypothesis_only())

                elif input_type == 'full-seq':
                    examples.append(data_raw.obs1_hyp_obs2())

        return examples

    def get_labels(self, input_file: str):
        """Reads the labels from lst file."""
        labels = []
        with open(input_file, "rb") as f:
            for l in f:
                labels.append(l.decode().strip())

        return labels

    def create_binary_examples(self, file_path, input_type, labels_path=None):
        """Creates examples for the baseline model."""
        if labels_path != None:
            examples = self.create_example(FileReader.read_jsonl(file_path), input_type, self.get_labels(labels_path))
        else:
            examples = self.create_example(FileReader.read_jsonl(file_path))

        binary_examples = []
        binary_labels = []

        for example in examples:
            if int(example['label']) == 1:
                binary_examples.append(example['hyp1'])
                binary_labels.append(1)
                binary_examples.append(example['hyp2'])
                binary_labels.append(0)
            elif int(example['label']) == 2:
                binary_examples.append(example['hyp2'])
                binary_labels.append(1)
                binary_examples.append(example['hyp1'])
                binary_labels.append(0)

        return binary_examples, binary_labels


class Anli(Dataset):
    """
    Anli dataset: Inherits from pytorch Dataset module
    Returns: Training instance given index
    """

    def __init__(self, instances):
        self.config = omegaconf.OmegaConf.load("config.yaml")
        self.data = self._load_data(self.config['train_x'], self.config["train_y"])

        self.instances = instances

    def _load_data(self, x_path, y_path=None):
        df = pd.read_json(x_path, lines=True)
        if y_path:
            labels = pd.read_csv(y_path, sep='\t', header=None).values.tolist()
            self.label_offset = np.asarray(labels).min()
            df["label"] = np.asarray(labels) - self.label_offset

        return df

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        return self.instances[idx]
