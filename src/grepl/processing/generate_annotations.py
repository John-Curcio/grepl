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
import pandas as pd
import pickle
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
# TODO move this CSV_PATH to a config/constants file
CSV_PATH = "/home/rocus/Documents/john/grepl/parsed_page_enriched.csv"
CLIP_FRAMES_FOLDER = "/home/rocus/Documents/john/grepl/clip_frames"
LABELED_TAGS_FILENAME = "/home/rocus/Documents/john/grepl/labeled_tags.txt" # each tag is "labeled" as "bjj" or "other"
ANNOTATION_FILENAME = "annotations.txt"
ANNOTATION_CSV_FILENAME = "annotations.csv"

def _get_clip_filename(row, duration: int = 30, bw: bool = False) -> str:
    """Get the filename, as it would be in the clip_frames folder. Don't include the extension."""
    vid, start = row["youtube_id"], row["total_seconds"]
    return f"{vid}_{start}_{duration}{'_bw' if bw else ''}"

def _fit_label_encoder(df: pd.DataFrame) -> LabelEncoder:
    le = LabelEncoder()
    all_tags = sorted({tag for tags_str in df["tags"] for tag in tags_str.split(",")})

    # all_tags = sorted(set(df["tags"].apply(lambda x: x.split(",")).explode()))
    
    le.fit(all_tags)
    return le

def generate_annotations():
    """Generate the clip_frames/annotations.txt folder (and the label encoder to make sense of the integer labels)
    
    Each row of annotations.txt will look like this:
    clip_filename 0 num_frames i j k ...
    
    where i, j, k are the integer labels for the tags associated with the clip.
    Uses the labeled tags dataset to say whether a tag is relevant to learning BJJ or not (ie filter out
    all the proper nouns, like "John Danaher" or "Gordon Ryan" or "BJJ Fanatics" etc.).
    
    TODO maybe I can refactor this into a dataclass?"""
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
    df["encoded_tags_exhaustive"] = df["tags"].apply(lambda x: le.transform(x.split(",")))

    print("Encoding tags worth learning...")
    labeled_tags = pd.read_csv(LABELED_TAGS_FILENAME, header=None, names=["tag", "label"])
    bjj_tags = set(labeled_tags.query("label == 'bjj'")["tag"])
    df["tags_filtered"] = df["tags"].apply(lambda x: ",".join(
        [t for t in x.split(",") if t in bjj_tags]
    ))
    df["encoded_tags_filtered"] = df["tags_filtered"].apply(lambda x: le.transform(x.split(",")) if x else [])

    # save the annotations CSV
    df.to_csv(os.path.join(CLIP_FRAMES_FOLDER, ANNOTATION_CSV_FILENAME), index=False)

    # save the label encoder
    le_filepath = os.path.join(CLIP_FRAMES_FOLDER, "label_encoder.pkl")
    with open(le_filepath, "wb") as f:
        pickle.dump(le, f)
    
    # create the annotations.txt file
    if not os.path.exists(CLIP_FRAMES_FOLDER):
        os.makedirs(CLIP_FRAMES_FOLDER)
    annotation_filepath = os.path.join(CLIP_FRAMES_FOLDER, ANNOTATION_FILENAME)
    with open(annotation_filepath, "w") as f:
        n_rows = df.query("clip_filename_in_clips_folder").shape[0]
        for _, row in tqdm(df.query("clip_filename_in_clips_folder").iterrows(), total=n_rows):
            label_annotation = " ".join([str(t) for t in row["encoded_tags_exhaustive"]])
            clip_filename = row["clip_filename"]
            # count number of images in clip_frames folder
            num_images = len(os.listdir(os.path.join(CLIP_FRAMES_FOLDER, clip_filename)))
            annotation_row = f"{clip_filename} 0 {num_images} {label_annotation}"
            f.write(annotation_row + "\n")

if __name__ == "__main__":
    generate_annotations()