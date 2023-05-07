import os
import shutil
from features import FeatureExtractor
from dataset_reader import SRDatasetReader, SRCodecsReader, MCLReader, QADSReader


feats = FeatureExtractor([
                        "erqa",
                        "SI",
                        "TI",
                        "ssim", 
                        "lpips", 
                        "gabor",
                        "sobel",
                        "lbp",
                        "haff",
                        "sobel_sd",
                        "optical",
                        "fft",
                        "laplac",
                        "colorfulness_diff",
                        "colorfulness",
                        "hist"
                    ])


def write(feats, path):
    if not os.path.exists(path):
        line = ""
        with open(path, 'w') as f:
            for key in feats:
                line += key + "\t"
            line = line[:-1] + "\n"
            f.write(line)
    shutil.copyfile(path, path[:-4] + "_copy.tsv")

    line = ""
    for key in feats:
        value = str(feats[key]) + "\t"
        line += value
    line = line[:-1] + "\n"
    with open(path, 'a') as f:
        f.write(line)    
    

def check_exist(info, file_path):
    if not os.path.exists(file_path):
        return False
    with open(file_path, 'r') as f:
        lines = list(f.readlines())

    key = info["gt"] + "@" + info["dist"]
    for line in lines:
        if line.startswith(key):
            return True
    return False


def scrap_feats(dataset, feature_extractor, file_path):
    length = len(dataset)
    for i, elem in enumerate(dataset.iterator()):
        print(f"Video {i  + 1} / {length}")
        dist, gt, score, info = elem
        if check_exist(info, file_path):
            continue
        features = feature_extractor(dist, gt)
        feature_extractor.reinit()
        
        new_features = {
            "info" : info["gt"] + "@" + info["dist"]
        }
        for ft in features:
            for key in features[ft]:
                new_ft_name = ft + "_" + key
                new_features[new_ft_name] = features[ft][key]
        
        new_features["subjective"] = score

        write(new_features, file_path)
    