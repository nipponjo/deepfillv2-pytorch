# %%
import pathlib
import gdown

# %%


FILES_DICT = {  
    # converted weights: Places2 (for networks_tf.py)
    "states_tf_places2.pth": {
        "path": "pretrained/states_tf_places2.pth",
        "url": "https://drive.google.com/file/d/1tvdQRmkphJK7FYveNAKSMWC6K09hJoyt/view?usp=drive_link",
        "download": True,
    },
    # converted weights: CelebA-HQ (for networks_tf.py)
    "states_tf_celebahq.pth": {
        "path": "pretrained/states_tf_celebahq.pth",
        "url": "https://drive.google.com/file/d/1fTQVSKWwWcKYnmeemxKWImhVtFQpESmm/view?usp=drive_link",
        "download": True,
    },
    # fine-tuned weights: Places2 (for networks.py)
    "states_pt_places2.pth": {
        "path": "pretrained/states_pt_places2.pth",
        "url": "https://drive.google.com/file/d/1L63oBNVgz7xSb_3hGbUdkYW1IuRgMkCa/view?usp=drive_link",
        "download": True,
    },
    # fine-tuned weights: CelebA-HQ (for networks.py)
    "states_pt_celebahq.pth": {
        "path": "pretrained/states_pt_celebahq.pth",
        "url": "https://drive.google.com/file/d/17oJ1dJ9O3hkl2pnl8l2PtNVf2WhSDtB7/view?usp=drive_link",
        "download": True,
    },

}

# %%

root_dir = pathlib.Path(__file__).parent

for file_dict in FILES_DICT.values():
    file_path = root_dir.joinpath(file_dict['path'])

    if not file_path.parent.exists():
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
    if file_path.exists():
        print(file_dict['path'], "already exists!")
    elif file_dict.get('download', True):
        print("Downloading", file_dict['path'], "...")
        output_filepath = gdown.download(file_dict['url'], output=file_path.as_posix(), fuzzy=True)
