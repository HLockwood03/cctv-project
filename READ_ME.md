# CCTV Fight Detector

## Introduction
This is the repository for the CCTV Fight Detector, my Part III Project at the University of Southampton.

This project builds off of the work of ST-GCN; a type of Graph Convolutional Network that incorporates both spatial and temporal features of a video to classify it.

PLEASE NOTE: the data of the UBI FIGHTS, UCF Crime, Movies, Kinetics and Fight-Detection datasets are not included with the submission as they are too big. Please ensure if you are missing them and need them to download!

## Prerequisites
To use the CCTV Fight Detector, you must have OpenPose and ST-GCN configured on your PC.
This repository comes with them included.
OpenPose's Python API is required to run this program.
The OpenPose Python API must be located in `PYTHONPATH`.
- Python3 (developed with 3.8.8 specifically, not tested on other versions!)
    - Pytorch
    - pyyaml
    - argparse
    - numpy
    - h5py
    - opencv-python
    - imageio
    - scikit-learn
    - scikit-video
    - torch (2.1.0 with cuda 12.1)
    - torchinfo
    - torchlight (run setup.py in st_gcn/torchlight/)
    - torchvision (0.16.0 with cuda 12.1)
- CUDA 12.2
- CUDNN 8.9.4
- PYTHONPATH Environment Variable with paths to:
    -Python38 (or whatever your version is)
    -Python38\Lib
    -Python38\DLLs
    -openpose\build\python\openpose\Debug
- Path Environment Variable with paths to:
    -Python38
    -Python38\Scripts
    -Python38\Library\bin


## Demo
To use the CCTV Fight Detector, run in CMD with the following command:envi
```shell
python st_gcn\main.py demo_fight --video camera_source
```

If you wish to test it on a video located on your PC, run it as such:
```shell
python st_gcn\main.py demo_fight --video {PATH_TO_VIDEO}
```

## Testing
A large-scale test functionality is available. This supports both temporal-level and video-level testing, temporal level giving the ground truth classification for every frame in a video, whereas video-level gives the ground truth classification for the whole video.
Run as such:
```shell
python st_gcn\main.py demo_fight --mode test_video
```

```shell
python st_gcn\main.py demo_fight --mode test_temporal
```

Ensure the paths in demo_fight.yaml lead to the appropriate folders/files:
- annotation_path: 
    - For Temporal Level, links to each of the .csv files for every video containing 0s for frames with no fight and 1s for frames with fight.
- video_annotation_path: 
    - For Video Level, links to a .csv file which is line separated for every video, with name/classification comma separated on each line.
- video_path:
    - Directory for where the videos are stored
- test_split:
    - For Temporal Level; indicates which of the videos you wish to use.
- prediction_out:
    - For Temporal Level; indicates where the intermediary predictions/f1 scores should be stored.

The final F1 score and accuracy score will be printed to terminal once testing is complete.

## Making your own dataset
For the purposes of transfer learning, a functionality has been developed to use OpenPose's skeleton output feature to create the skeleton data needed.
Should you wish to do your own transfer learning, the following functionality is designed to assist this.

```shell
python skeleton.py
```
skeleton.py can be found under the OpenPose repository included.
Configure the variables in the main method to change outputs accordingly.
- videos_folder:
    - The input folder, for where the raw RGB data to use for transfer learning is held.
- skeleton_folder:
    - The output folder, where generated skeleton data will be placed.
- fightDir
    - Subfolder of videos_folder: where violent videos are held
- nonFightDir
    - Subfolder of videos_folder: where non-violent videos are held
- trainPath/testPath
    - The txt files containing the train/test split.

This command will output the following in skeleton_folder:
- {}_train_label.json/{}_val_label.json
    - json files containing the details of each skeleton piece in the train/val datasets
    - has skeleton, label, label index
    - currently has skeleton is always set to true
- {}_train/{}_val
    - Folders containing the json files for every video
    - JSON files describe every frame, and every skeleton in the frame, with the position/confidence for every point of the skeleton.

After this, the following command under st_gcn directory should be used:
```shell
python .\tools\fd_gendata.py
```

This will take the generated JSON files from skeleton.py and turn them into train_data.npy and val_data.npy.
These files are the types of file you will need to supply to transferprocessor.py.

## Transfer Learning
One can use these generated .npy files from the above section to facilitate transfer learning in transferprocessor.py.
To run it:
```shell
python main.py transfer -c .\config\fd\transfer.yaml 
```

The arguments of transfer.yaml can be changed to facilitiate if you are using differently named data.
- work_dir
    - The name of the output folder for where models are saved.
- train/test_feeder_args
    - Arguments for where train/test data is gotten from.
- train_full_feeder_args
    - Arguments for when running with cross_validation enabled.

The arguments of #optim section can be configured to hyperparameter tune.
The outputted .pt weights file can be used in the demo_fight.yaml, under weights as your new model for your new transfer learned purpose.