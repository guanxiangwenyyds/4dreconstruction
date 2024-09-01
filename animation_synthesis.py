import cv2
import os

# 图片文件夹路径


def animation_synthesis(image_folder, video_name):

    video_dir = os.path.join(image_folder, 'video')
    os.makedirs(video_dir, exist_ok=True)

    video_path = os.path.join(video_dir, f'{video_name}.mp4')

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort()

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_path, fourcc, 30, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))
    video.release()

    print(f'{video_name} saved at {video_path}')










