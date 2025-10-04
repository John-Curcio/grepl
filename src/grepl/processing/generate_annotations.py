"""
TODO:
see 2025_07_24_known_pos.ipynb for example...

* Load the parsed_page_enriched table from sqlite
* Get mp4 filename + tags
* Filter to clips present in the clip_frames folder
* Fit sklearn LabelEncoder on the tags
* Write annotations to clip_frames folder
* Save the LabelEncoder to disk
* call this as part of videos_to_frames.py?
TODO see docs for more: https://github.com/RaivoKoot/Video-Dataset-Loading-Pytorch/tree/main?tab=readme-ov-file#6-allowing-multiple-labels-per-sample
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from grepl.constants import (
    ANNOTATION_CSV_FILENAME,
    ANNOTATION_FILENAME,
    CLIP_FRAMES_DIR,
    FILTERED_LABEL_ENCODER_FILENAME,
    LABEL_ENCODER_FILENAME,
    LABELED_TAGS_FILE,
    PARSED_PAGE_CSV,
)

CSV_PATH = PARSED_PAGE_CSV
CLIP_FRAMES_FOLDER = CLIP_FRAMES_DIR
LABELED_TAGS_FILENAME = LABELED_TAGS_FILE  # each tag is "labeled" as "bjj" or "other"

P_TRAIN = 0.8
P_VALID, P_TEST = (1 - P_TRAIN) / 2, (1 - P_TRAIN) / 2
np.random.seed(42)  # for reproducibility

def _get_clip_filename(row, duration: int = 30, bw: bool = False) -> str:
    """Get the filename, as it would be in the clip_frames folder. Don't include the extension."""
    vid, start = row["youtube_id"], row["total_seconds"]
    return f"{vid}_{start}_{duration}{'_bw' if bw else ''}"

def _fit_label_encoder(df: pd.DataFrame, tags_col: str = "tags") -> LabelEncoder:
    le = LabelEncoder()
    all_tags = sorted({tag for tags_str in df[tags_col] for tag in tags_str.split(",")})
    le.fit(all_tags)
    return le

def _encode_tags_dummy(tags:str, le: LabelEncoder) -> np.ndarray:
    """Encode tags as a dummy vector using the provided label encoder.
    Not technically one-hot encoded because multiple tags can be present.
    """
    encoded_tags = le.transform(tags.split(",")) if tags else []
    dummy = np.zeros(len(le.classes_), dtype=int)
    dummy[encoded_tags] = 1
    return dummy

def _write_annotations_to_file(df: pd.DataFrame, annotation_filepath: str,
                               encoded_tags_col: str = "encoded_tags_exhaustive",
                               min_num_frames: int = 30):
    # create the annotations.txt file
    with open(annotation_filepath, "w") as f:
        n_rows = df.query("clip_filename_in_clips_folder").shape[0]
        skipped_clips = 0
        for _, row in tqdm(df.query("clip_filename_in_clips_folder").iterrows(), total=n_rows):
            label_annotation = " ".join([str(t) for t in row[encoded_tags_col]])
            clip_filename = row["clip_filename"]
            # count number of images in clip_frames folder
            num_images = len(set(
                f for f in os.listdir(os.path.join(CLIP_FRAMES_FOLDER, clip_filename))
                if f.endswith(".jpg") or f.endswith(".jpeg")
            ))
            
            # Skip clips that don't have enough frames
            if num_images < min_num_frames:
                skipped_clips += 1
                continue
                
            annotation_row = f"{clip_filename} 0 {num_images-1} {label_annotation}"
            f.write(annotation_row + "\n")
        
        if skipped_clips > 0:
            print(f"Skipped {skipped_clips} clips with fewer than {min_num_frames} frames")

def generate_annotations(min_num_frames: int = 30):
    """Generate the clip_frames/annotations.txt folder (and the label encoder to make sense of the integer labels)
    
    Each row of annotations.txt will look like this:
    clip_filename 0 num_frames i j k ...
    
    where i, j, k are the integer labels for the tags associated with the clip.
    Uses the labeled tags dataset to say whether a tag is relevant to learning BJJ or not (ie filter out
    all the proper nouns, like "John Danaher" or "Gordon Ryan" or "BJJ Fanatics" etc.).
    
    TODO maybe I can refactor this into a dataclass?
    """
    df = pd.read_csv(CSV_PATH)
    print("Loaded CSV with shape:", df.shape)
    df["clip_filename"] = df.apply(_get_clip_filename, axis=1)

    print("Checking if clips are present in the clip_frames folder...")
    df["clip_filepath_if_present"] = df["clip_filename"].apply(
        lambda x: os.path.join(CLIP_FRAMES_FOLDER, x)
    )
    df["clip_filename_in_clips_folder"] = df["clip_filepath_if_present"].apply(os.path.exists)
    print("Found", df["clip_filename_in_clips_folder"].sum(), "clips in the clip_frames folder.")
    print("Fitting, transforming label encoder on tags...")
    le = _fit_label_encoder(df)
    df["encoded_tags_exhaustive"] = df["tags"].apply(lambda x: _encode_tags_dummy(x, le))

    print("Encoding tags worth learning...")
    labeled_tags = pd.read_csv(LABELED_TAGS_FILENAME, header=None, names=["tag", "label"])
    bjj_tags = set(labeled_tags.query("label == 'bjj'")["tag"])
    df["tags_filtered"] = df["tags"].apply(lambda x: ",".join(
        [t for t in x.split(",") if t in bjj_tags]
    ))
    filtered_le = _fit_label_encoder(df, tags_col="tags_filtered")
    df["encoded_tags_filtered"] = df["tags_filtered"].apply(lambda x: _encode_tags_dummy(x, filtered_le))

    # save the label encoder
    for le, filename in [(le, LABEL_ENCODER_FILENAME), (filtered_le, FILTERED_LABEL_ENCODER_FILENAME)]:
        le_filepath = os.path.join(CLIP_FRAMES_FOLDER, filename)
        with open(le_filepath, "wb") as f:
            pickle.dump(le, f)
    # save the annotations.txt for the entire dataset
    print("Writing annotations to file... (full dataset)")
    if not os.path.exists(CLIP_FRAMES_FOLDER):
        os.makedirs(CLIP_FRAMES_FOLDER)
    annotation_filepath = os.path.join(CLIP_FRAMES_FOLDER, ANNOTATION_FILENAME)
    _write_annotations_to_file(df, annotation_filepath, min_num_frames=min_num_frames)
    # now, split the dataset into train, valid, test sets
    # grouped by youtube_id (no youtube_ids leakage)
    youtube_id_assignment = {
        youtube_id: np.random.choice(["train", "valid", "test"], 
                                     p=[P_TRAIN, P_VALID, P_TEST])
        for youtube_id in df["youtube_id"].unique()
    }
    df["split_assignment"] = df["youtube_id"].apply(
        lambda x: youtube_id_assignment[x]
    )
    for split_name, split_df in df.groupby("split_assignment"):
        for encoded_tags_suffix in ["exhaustive", "filtered"]:
            split_annotation_filepath = os.path.join(CLIP_FRAMES_FOLDER, 
                                                     f"annotations_{encoded_tags_suffix}_{split_name}.txt")
            print(f"Writing {len(split_df)} annotations to file... ({split_name} set, {encoded_tags_suffix} tags)")
            encoded_tags_col = f"encoded_tags_{encoded_tags_suffix}"
            _write_annotations_to_file(split_df, split_annotation_filepath, encoded_tags_col, min_num_frames)

    # save the annotations CSV
    df.to_csv(os.path.join(CLIP_FRAMES_FOLDER, ANNOTATION_CSV_FILENAME), index=False)

if __name__ == "__main__":
    generate_annotations()
