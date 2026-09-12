# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""
Preprocess the R2I-Bench dataset to parquet format for DualGRPO training.
You can obtain the raw dataset (xxx.csv) from:
https://github.com/PLUM-Lab/R2I-Bench/tree/main/data/prompts

Usage::

    python examples/dualgrpo_trainer/data_process/r2i_bench.py \
        --input_dir ~/data/r2i_bench/prompts/ \
        --output_dir ~/data/r2i_bench/qwen_image/
"""

import argparse
import os

from datasets import concatenate_datasets, load_dataset

category_subcategory_list = [
    {"category": "commonsense", "subcategory": "social_cultural_knowledge_object"},
    {"category": "commonsense", "subcategory": "temporal_understanding"},
    {"category": "commonsense", "subcategory": "social_cultural_knowledge_scene"},
    {"category": "compositional", "subcategory": "creative_compositional"},
    {"category": "compositional", "subcategory": "inferential_spatial"},
    {"category": "compositional", "subcategory": "prescriptive_spatial"},
    {"category": "concept_mixing", "subcategory": "functional_mixing"},
    {"category": "concept_mixing", "subcategory": "literal_mixing"},
    {"category": "logical", "subcategory": "abductive"},
    {"category": "logical", "subcategory": "categorical"},
    {"category": "logical", "subcategory": "conjunctive"},
    {"category": "logical", "subcategory": "deductive"},
    {"category": "logical", "subcategory": "disjunctive"},
    {"category": "logical", "subcategory": "hypothetical"},
    {"category": "logical", "subcategory": "sufficient_conditional"},
    {"category": "numerical", "subcategory": "approximate_number_generation"},
    {"category": "numerical", "subcategory": "conceptual_quantitative"},
    {"category": "numerical", "subcategory": "exact_number_generation"},
    {"category": "mathematical", "subcategory": "combinatorial"},
    {"category": "mathematical", "subcategory": "cryptographic_encoding"},
    {"category": "mathematical", "subcategory": "geometrical_transformations"},
    # {"category": "mathematical", "subcategory": "mathematical_function"},
    {"category": "mathematical", "subcategory": "number_theory"},
    {"category": "mathematical", "subcategory": "spatial_reasoning"},
    {"category": "mathematical", "subcategory": "vector_matrix_visualization"},
    {"category": "mathematical", "subcategory": "set_theory"},
    {"category": "causal", "subcategory": "cause_to_effect"},
    {"category": "causal", "subcategory": "effect_to_cause"},
]

# system prompt used in DualGRPO
SYSTEM_PROMPT = (
    "You are a Prompt Optimizer specializing in image generation models (e.g., MidJourney, Stable Diffusion)."
    " Your core task is to rewrite user-provided prompts into highly clear, easy-to-render versions.\n"
    "When rewriting, prioritize the following principles:\n"
    "1. Start from the user's prompt, do reasoning step by step to analyze the object or scene they want to generate.\n"
    "2. Focus on describing the final visual appearance of the scene. "
    "Clarify elements like the main subject’s shape, color, and state.\n"
    "3. If you are confident about what the user wants to generate, "
    "directly point it out in your explanation and the final revised prompt.\n"
    "4. If technical concepts are necessary but difficult for ordinary users to understand, "
    "translate them into intuitive visual descriptions.\n"
    "5. Ensure the final revised prompt is consistent with the user's intent.\n\n"
    "After receiving the user’s prompt that needs rewriting, first explain your reasoning for optimization. "
    'Then, output the final revised prompt in the fixed format of "Revised Prompt:\n". '
    "Where the specific revised content is filled in the next line.\n\n"
    "Prompt:\n"
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess R2I-Bench dataset for DualGRPO training.")
    parser.add_argument(
        "--input_dir",
        default="~/data/r2i_bench/prompts/",
        help="Path to the raw dataset directory (contains */*.csv).",
    )
    parser.add_argument(
        "--output_dir",
        default="~/data/r2i_bench/qwen_image/",
        help="Directory to save the preprocessed parquet files.",
    )
    parser.add_argument(
        "--shuffle",
        default=False,
        action="store_true",
        help="Whether to shuffle data when spliting data.",
    )

    args = parser.parse_args()
    local_dataset_path = os.path.expanduser(args.input_dir)
    if local_dataset_path is None:
        raise NotImplementedError(
            "It is not existed in huggingface hub. "
            "Please get dataset from https://github.com/PLUM-Lab/R2I-Bench/tree/main/data/prompts"
        )

    # load all csv files
    train_datasets_to_merge = []
    test_datasets_to_merge = []

    for entry in category_subcategory_list:
        category = entry["category"]
        subcategory = entry["subcategory"]
        filename = f"{category}_{subcategory}.csv"
        csv_path = os.path.join(local_dataset_path, category, filename)
        if not os.path.exists(csv_path):
            print(f"File does not exist: {csv_path}")
            continue
        sub_dataset = load_dataset("csv", data_files=csv_path)

        # make sure split with same ratio for each subcategory
        # deterministic split with shuffle=False (easy for debug)
        # split to train and test set with ratio 8:2
        split_dataset = sub_dataset["train"].train_test_split(test_size=0.2, shuffle=args.shuffle)
        train_datasets_to_merge.append(split_dataset["train"])
        test_datasets_to_merge.append(split_dataset["test"])

    # Merge the train and test sets
    train_dataset = concatenate_datasets(train_datasets_to_merge)
    test_dataset = concatenate_datasets(test_datasets_to_merge)

    # Prepare data format
    data_source = "r2i_bench"
    system_prompt = SYSTEM_PROMPT
    negative_user_prompt = " "

    def make_map_fn(split):
        def process_fn(example, idx):
            raw_prompt = example.pop("Prompt")
            data = {
                "data_source": data_source,
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": raw_prompt},
                ],
                "negative_prompt": [
                    {"role": "user", "content": negative_user_prompt},
                ],
                "ability": "preference_alignment",
                "reward_model": {"style": "model", "ground_truth": raw_prompt},
                "extra_info": {"split": split, "index": idx},
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_save_dir = args.output_dir

    local_save_dir = os.path.expanduser(local_save_dir)
    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))
