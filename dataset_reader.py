import os
import json
from pathlib import Path
import random
import subprocess
import cv2


class DatasetReader:
    def __init__(self, path : Path):
        self.path = Path(path)

        self.gt_path = None
        self.dist_path = None
        self.scores_path = None
        
        self.video2scores = {}
        self.video2path = {}
        self.video2gt = {}

    def get_random_video(self):
        gt = random.choice(list(self.video2scores.keys()))
        pair_name = random.choice(list(self.video2scores[gt].keys()))

        video_path = self.video2path[gt][pair_name]
        gt_path = self.video2gt[gt][pair_name]
        score = self.video2scores[gt][pair_name]

        return (self.get_frames(video_path), self.get_frames(gt_path), score, {"gt" : gt, "dist" : pair_name})
    
    def iterator(self):
        for gt in self.video2path.keys():
            for pair_name in self.video2path[gt].keys():
                video_path = self.video2path[gt][pair_name]
                gt_path = self.video2gt[gt][pair_name]
                score = self.video2scores[gt][pair_name]

                yield (self.get_frames(video_path), self.get_frames(gt_path), score, {"gt" : gt, "dist" : pair_name})

    def get_frames(self, path : Path):
        if os.path.isdir(path):
            frames_paths = os.listdir(path)
            frames = []
            for f_path in frames_paths:
                frames.append(cv2.imread(str(path) + "/" + f_path))
            return frames
        else:
            return [cv2.imread(str(path))]
    
    def __len__(self):
        length = 0
        for gt in self.video2path.keys():
            for __ in self.video2path[gt].keys():
                length += 1
        return length  

class SRDatasetReader(DatasetReader):

    def __init__(self, path : Path):
        super().__init__(path)

        self.gt_path = self.path / "gt"
        self.dist_path = self.path / "videos"
        self.scores_path = self.path / "scores"

        self.video2scores = {}
        self.video2path = {}
        self.video2gt = {}

        for file in os.listdir(self.scores_path):
            with open(self.scores_path / file, 'r') as f:
                lines = list(f.readlines())
            
            gt_name = file[len("name="):].split("_")[0]
            if gt_name in ["constructor", "seesaw"]:
                continue

            self.video2scores[gt_name] = {}
            self.video2path[gt_name] = {}
            self.video2gt[gt_name] = {}

            for i, line in enumerate(lines):
                if i == 0:
                    continue
            
                words = line[:-1].split(";")
                pair_name = words[0][1:-1]
                score = float(words[1].replace(",", "."))

                self.video2scores[gt_name][pair_name] = score
                self.video2path[gt_name][pair_name] = self.dist_path / gt_name / pair_name
                self.video2gt[gt_name][pair_name] = self.gt_path / gt_name / "frames"

    

class SRCodecsReader(DatasetReader):

    def __init__(self, path : Path):
        super().__init__(path)

        self.gt_path = self.path / "gt"
        self.dist_path = self.path / "videos"
        self.scores_path = self.path / "scores"

        self.video2scores = {}
        self.video2path = {}
        self.video2gt = {}

        for codec in os.listdir(self.scores_path):
            for file in os.listdir(self.scores_path / codec):
                with open(self.scores_path / codec / file, 'r') as f:
                    lines = list(f.readlines())
                
                gt_name = file[len("name="):].split("_")[0]

                if gt_name not in self.video2gt.keys():
                    self.video2scores[gt_name] = {}
                    self.video2path[gt_name] = {}
                    self.video2gt[gt_name] = {}

                for i, line in enumerate(lines):
                    if i == 0:
                        continue
                
                    words = line[:-1].split(";")
                    pair_name = codec + "_" + words[0][1:-1]
                    score = float(words[1].replace(",", "."))

                    self.video2scores[gt_name][pair_name] = score
                    self.video2path[gt_name][pair_name] = self.dist_path / codec / gt_name / words[0][1:-1]
                    self.video2gt[gt_name][pair_name] = self.gt_path / gt_name


class MCLReader(DatasetReader):

    def __init__(self, path : Path):
        super().__init__(path)

        self.gt_path = self.path / "reference_YUV_sequences"
        self.dist_path = self.path / "distorted_YUV_sequences"
        self.scores_path = self.path / "opinion_scores"

        self.video2scores = {}
        self.video2path = {}
        self.video2gt = {}

        f = open(self.scores_path / "mean_opinion_scores.txt", 'r')
        lines = list(f.readlines())
        f.close()

        for line in lines:
            words = line[:-1].split("\t")
            video_name = words[0]
            if video_name == "DanceKiss_H264_4":
                continue
            score = float(words[1])
            words = video_name.split("_")
            gt_name = words[0]

            if gt_name not in self.video2gt.keys():
                self.video2scores[gt_name] = {}
                self.video2path[gt_name] = {}
                self.video2gt[gt_name] = {}

            self.video2scores[gt_name][video_name] = score
            self.video2path[gt_name][video_name] = self.dist_path / video_name
            self.video2gt[gt_name][video_name] = self.gt_path / gt_name


class QADSReader(DatasetReader):

    def __init__(self, path : Path):
        super().__init__(path)

        self.gt_path = self.path / "source_images"
        self.dist_path = self.path / "super-resolved_images"
        self.scores_path = self.path / "mos_with_names.txt"

        self.video2scores = {}
        self.video2path = {}
        self.video2gt = {}

        f = open(self.scores_path, 'r')
        lines = list(f.readlines())
        f.close()

        for line in lines:
            words = line[:-1].split("  ")
            video_name = words[1]
            score = float(words[0])
            words = video_name.split("_")
            gt_name = words[0] + ".bmp"

            if gt_name not in self.video2gt.keys():
                self.video2scores[gt_name] = {}
                self.video2path[gt_name] = {}
                self.video2gt[gt_name] = {}

            self.video2scores[gt_name][video_name] = score
            self.video2path[gt_name][video_name] = self.dist_path / video_name
            self.video2gt[gt_name][video_name] = self.gt_path / gt_name
