import os
import argparse

import cv2
#import wget
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--output_path', type=str, default='data/LSMDC-E/videos', help='directory to output frames')
    parser.add_argument('--user_name', type=str, default='', help='your user name to LSMDC')
    parser.add_argument('--password', type=str, default='', help='your password to LSMDC')

    args = parser.parse_args()        

    with open("download_video_urls.txt") as f:
        for line in f.readlines():
            url = line.strip('\n').strip()
            #wget.download(url)
            subprocess.call(['wget', url, '--user=' + args.user_name, '--password='+args.password])

            video_file = url.split('/')[-1]
            video_name = '.'.join(video_file.split('.')[:-1])
            vc = cv2.VideoCapture(video_file)

            folder = os.path.join(args.output_path, video_name)
            os.makedirs(folder)
            fps = int(vc.get(cv2.CAP_PROP_FPS))

            valid = True
            count = 0
            while(valid):
                valid, frame = vc.read()
                if valid and count % fps == 0:
                    cv2.imwrite(os.path.join(folder, '{}.jpg'.format(int(count / fps))), frame)
                count += 1

            vc.release()
            os.remove(video_file)
