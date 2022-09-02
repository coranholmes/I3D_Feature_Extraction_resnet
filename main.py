from pathlib import Path
import shutil
import argparse
import numpy as np
import time
import ffmpeg
from resnet import i3_res50
import os


def generate(args):
    datasetpath, outputpath, pretrainedpath, frequency, batch_size, sample_mode, last_segment, format, file_type, version = args.datasetpath, args.outputpath, args.pretrainedpath, args.frequency, args.batch_size, args.sample_mode, args.last_segment, args.video_format, args.file_type, args.version
    if version == "v2":
        from extract_features import run
    elif version == "v3":
        from extract_features_weiling import run

    Path(outputpath).mkdir(parents=True, exist_ok=True)
    temppath = outputpath + "/temp/"
    rootdir = Path(datasetpath)

    if args.file_type == "video":
        videos = [str(f) for f in rootdir.glob('**/*.' + format)]
    elif args.file_type == "image":
        videos = os.listdir(rootdir)
    else:
        raise Exception("file_type must be video or image")

    # setup the model
    i3d = i3_res50(400, pretrainedpath)
    i3d.cuda()
    i3d.train(False)  # Set model to evaluate mode
    for video in videos:
        videoname = video.split("/")[-1].split(".")[0]
        feat_save_path = outputpath + "/" + videoname + "_i3d.npy"
        print(feat_save_path)
        if os.path.isfile(feat_save_path):
            print("{} already exists".format(feat_save_path))
        else:
            startime = time.time()
            print("Generating for {0}".format(video))
            Path(temppath).mkdir(parents=True, exist_ok=True)
            if args.file_type == "video":
                ffmpeg.input(video).output('{}%d.jpg'.format(temppath), start_number=0).global_args('-loglevel', 'quiet').run()
            elif args.file_type == "image":
                # copy images to temp folder
                frames_path = os.path.join(rootdir, video)
                for img in os.listdir(frames_path):
                    shutil.copy(os.path.join(frames_path, img), temppath)
            else:
                raise Exception("file_type must be video or image")

            print("Preprocessing done..")
            features = run(i3d, frequency, temppath, batch_size, sample_mode, last_segment)
            np.save(feat_save_path, features)
            shutil.rmtree(temppath)
            print("Obtained features of size: ", features.shape)
            print("done in {0}.".format(time.time() - startime))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasetpath', type=str, default="samplevideos/")
    parser.add_argument('--outputpath', type=str, default="output2")
    parser.add_argument('--pretrainedpath', type=str, default="pretrained/i3d_r50_kinetics.pth")
    parser.add_argument('--frequency', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--sample_mode', type=str, default="oversample")
    parser.add_argument('--last_segment', type=str, default="padding", choices={"padding", "cutting"},
                        help="whether to pad or cut the last segment")
    parser.add_argument('--video_format', type=str, default="mp4", help="video format")
    parser.add_argument('--file_type', type=str, default="video", choices=['video', 'image'], help="file type")
    parser.add_argument('--version', type=str, choices=["v2", "v3"], default="v3", help="version")
    args = parser.parse_args()
    generate(args)
