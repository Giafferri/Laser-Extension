"""Load and label the FEVER splits used by the extension study."""

from datasets import load_dataset
from dataset_utils.abstract_dataset import AbstractDataset
from fever_experiment_utils import (
    FEVER_FILTERED_TOTAL_SIZE,
    FEVER_LASER_DEV_SIZE,
    FEVER_SPLIT_COMBINED,
    FEVER_SPLIT_LASER_DEV,
    FEVER_SPLIT_LASER_TEST,
    FEVER_SPLIT_PAPER_DEV,
    FEVER_SPLIT_PAPER_TEST,
)


class FEVER(AbstractDataset):

    def __init__(self):
        super(AbstractDataset, self).__init__()
        self.metadata = {}

    @staticmethod
    def _get_consistent_unique(dataset_split, split_name, global_offset):
        dp_claim_dict = dict()
        for local_ix, dp in enumerate(dataset_split):
            claim = dp["claim"]
            label = dp["label"]

            if claim not in dp_claim_dict:
                dp_claim_dict[claim] = {
                    "labels": {label},
                    "first_local_ix": local_ix,
                }
            else:
                dp_claim_dict[claim]["labels"].add(label)

        consistent = []
        split_local_ix = 0
        for claim, metadata in dp_claim_dict.items():
            labels = metadata["labels"]
            if len(labels) != 1:
                continue
            consistent.append(
                {
                    "question": claim,
                    "answer": list(labels)[0],
                    "paper_split": split_name,
                    "paper_split_local_ix": split_local_ix,
                    "global_ix": global_offset + split_local_ix,
                    "raw_first_local_ix": metadata["first_local_ix"],
                }
            )
            split_local_ix += 1

        return consistent

    @staticmethod
    def _with_evaluation_split(rows, split_name):
        tagged = []
        for local_ix, row in enumerate(rows):
            tagged_row = dict(row)
            tagged_row["evaluation_split"] = split_name
            tagged_row["evaluation_split_local_ix"] = local_ix
            tagged.append(tagged_row)
        return tagged

    def get_splits(self, logger):

        dataset = load_dataset("EleutherAI/fever",'v1.0')

        paper_dev = dataset[FEVER_SPLIT_PAPER_DEV]
        paper_test = dataset[FEVER_SPLIT_PAPER_TEST]
        self.metadata = {
            "dataset/name": "EleutherAI/fever",
            "dataset/config": "v1.0",
            "dataset/paper_dev_fingerprint": getattr(paper_dev, "_fingerprint", None),
            "dataset/paper_test_fingerprint": getattr(paper_test, "_fingerprint", None),
        }

        # Verify that the source splits do not share claims.
        claims_dev = [dp["claim"] for dp in paper_dev]
        claims_test = [dp["claim"] for dp in paper_test]

        logger.log(f"Raw paper_dev set is {len(claims_dev)} and paper_test set is {len(claims_test)}.")

        assert len(set(claims_dev).intersection(set(claims_test))) == 0, "dev and test set cannot share claims"
        logger.log("Paper_dev and paper_test splits dont have a common context/claim.")

        # Keep one row for each claim with a consistent label.
        dataset_dev = self._get_consistent_unique(
            paper_dev,
            split_name=FEVER_SPLIT_PAPER_DEV,
            global_offset=0,
        )
        dataset_test = self._get_consistent_unique(
            paper_test,
            split_name=FEVER_SPLIT_PAPER_TEST,
            global_offset=len(dataset_dev),
        )

        logger.log(f"After filtering paper_dev set is {len(dataset_dev)} and paper_test set is {len(dataset_test)}.")
        combined = dataset_dev + dataset_test
        if len(combined) != FEVER_FILTERED_TOTAL_SIZE:
            raise ValueError(
                f"Expected {FEVER_FILTERED_TOTAL_SIZE} filtered FEVER claims, found {len(combined)}. "
                "The upstream dataset or preprocessing order may have changed."
            )

        # Match the paper's 20/80 intervention-selection protocol.
        laser_dev = self._with_evaluation_split(combined[:FEVER_LASER_DEV_SIZE], FEVER_SPLIT_LASER_DEV)
        laser_test = self._with_evaluation_split(combined[FEVER_LASER_DEV_SIZE:], FEVER_SPLIT_LASER_TEST)
        paper_dev_tagged = self._with_evaluation_split(dataset_dev, FEVER_SPLIT_PAPER_DEV)
        paper_test_tagged = self._with_evaluation_split(dataset_test, FEVER_SPLIT_PAPER_TEST)
        combined_tagged = self._with_evaluation_split(combined, FEVER_SPLIT_COMBINED)

        logger.log(
            f"Paper-aligned LASER 20/80 split is {len(laser_dev)} validation claims "
            f"and {len(laser_test)} test claims."
        )
        return {
            FEVER_SPLIT_PAPER_DEV: paper_dev_tagged,
            FEVER_SPLIT_PAPER_TEST: paper_test_tagged,
            FEVER_SPLIT_COMBINED: combined_tagged,
            FEVER_SPLIT_LASER_DEV: laser_dev,
            FEVER_SPLIT_LASER_TEST: laser_test,
        }

    def get_dataset(self, logger, split=FEVER_SPLIT_COMBINED):
        splits = self.get_splits(logger)
        if split not in splits:
            available = ", ".join(sorted(splits))
            raise KeyError(f"Unknown FEVER split '{split}'. Available splits: {available}")

        dataset = splits[split]
        logger.log(
            f"Read FEVER split {split} of size {len(dataset)} "
            f"(paper_dev={len(splits[FEVER_SPLIT_PAPER_DEV])}, paper_test={len(splits[FEVER_SPLIT_PAPER_TEST])})."
        )
        return dataset
