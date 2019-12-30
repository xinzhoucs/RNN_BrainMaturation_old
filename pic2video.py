if __name__ == '__main__':
    import argparse
    import cv2
    import os
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--framedir', type=str, default='./variance_frame')#add by yichen
    parser.add_argument('--videodir', type=str, default='./variance_video')#add by yichen
    parser.add_argument('--videoname', type=str, default='Randodrd_ALLNEW256_fuse_onehot_input_variance_5fps.avi')#add by yichen
    parser.add_argument('--fps', type=int, default='5')#add by yichen
    args = parser.parse_args()

    video_dir = args.videodir + '/' + args.videoname
    framedir = args.framedir + '/'
    fps = args.fps


    path_list=os.listdir(framedir)
    path_list.sort(key= lambda x:int(x[:-4]))
    frame = cv2.imread(framedir+path_list[0])
    width, height = frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc('M','J','P','G') #opencv3.0
    videoWriter = cv2.VideoWriter(video_dir, fourcc, fps, (height, width))

    for filename in path_list:
        frame = cv2.imread(framedir+filename)
        videoWriter.write(frame)
        cv2.waitKey(1)

    videoWriter.release()