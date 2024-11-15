# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os

import datasets

_DESCRIPTION = """\
The `tldr_news` dataset was constructed by collecting a daily tech newsletter (available at 
https://tldr.tech/newsletter). Then for every piece of news, the "headline" and its corresponding "content" were 
collected. Such a dataset can be used to train a model to generate a headline from a input piece of text.
"""

# TODO: Add link to the official dataset URLs here
# The HuggingFace Datasets library doesn't host the datasets but only points to the original files.
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_URLS = {"all": "https://github.com/JulesBelveze/tldr_news/blob/main/1.3.0.tar.gz?raw=true"}


class TLDRNewsConfig(datasets.BuilderConfig):
    """BuilderConfig for TLDRNews."""

    def __init__(self, **kwargs):
        """BuilderConfig for TLDRNews.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(TLDRNewsConfig, self).__init__(**kwargs)


class TLDRNewsDataset(datasets.GeneratorBasedBuilder):
    """Dataset containing headline & content of pieces of news from the tldr tech newsletter."""

    VERSION = datasets.Version("1.2.0")

    BUILDER_CONFIGS = [
        TLDRNewsConfig(name="all", version=VERSION, description="This contains all the existing newsletter"),
    ]

    DEFAULT_CONFIG_NAME = "all"

    def _info(self):
        features = datasets.Features(
            {
                "headline": datasets.Value("string"),
                "content": datasets.Value("string"),
                "category": datasets.ClassLabel(
                    num_classes=5,
                    names=['Sponsor', 'Big Tech & Startups', 'Science and Futuristic Technology',
                           'Programming, Design & Data Science', 'Miscellaneous']
                )
            }
        )

        return datasets.DatasetInfo(description=_DESCRIPTION, features=features)

    def _split_generators(self, dl_manager):
        urls = _URLS[self.config.name]
  
        data_dir = dl_manager.download_and_extract(urls)
        data_dir = os.path.join(data_dir, str(self.config.version))
        self._data_dir = data_dir
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "train.json"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": os.path.join(data_dir, "test.json"), "split": "test"},
            ),
        ]

    def _generate_examples(self, filepath, split):
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        for key, row in enumerate(data):
            yield key, {"headline": row["headline"], "content": row["content"], "category": row["category"]}

    def save_to_json(self, output_dir):
        """Save the dataset to JSON files locally.
        
        Args:
            output_dir (str): Directory path where to save the JSON files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate and save train split
        train_data = []
        train_gen = self._generate_examples(
            os.path.join(self._data_dir, "train.json"), "train"
        )
        for _, example in train_gen:
            train_data.append(example)
            
        with open(os.path.join(output_dir, "train.json"), "w", encoding="utf-8") as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
            
        # Generate and save test split
        test_data = []
        test_gen = self._generate_examples(
            os.path.join(self._data_dir, "test.json"), "test"
        )
        for _, example in test_gen:
            test_data.append(example)
            
        with open(os.path.join(output_dir, "test.json"), "w", encoding="utf-8") as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)

        print(f"Dataset saved to {output_dir}")

# Usage example:
if __name__ == "__main__":
    dataset = TLDRNewsDataset()
    dataset.download_and_prepare()
    dataset.save_to_json("./local_dataset")

