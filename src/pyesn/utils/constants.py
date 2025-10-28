"""Project-wide constants (no imports required)."""

TRAIN_RECORD_FILE = "train_record.jsonl"
PREDICT_RECORD_FILE = "predict_record.jsonl"

TARGET_SERIES_KEY = "target_series"
OUTPUT_SERIES_KEY = "output_series"

NUM_OF_CLASS = 3

NUM_TO_BEHAVIOR = {
    0: 'other',
    1: 'foraging',
    2: 'rumination'
}

BEHAVIOR_TO_NUM = {
    'other': 0,
    'foraging': 1,
    'rumination': 2
}