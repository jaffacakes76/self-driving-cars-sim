from moviepy.editor import ImageSequenceClip
import os

IMAGE_EXT = ['jpeg', 'gif', 'png', 'jpg']


def main():
    image_folder = "./images"
    fps = 45

    image_list = sorted([os.path.join(image_folder, image_file) for image_file in os.listdir(image_folder)])
    image_list = [image_file for image_file in image_list if os.path.splitext(image_file)[1][1:].lower() in IMAGE_EXT]

    video_file = 'output_video.mp4'

    print("Creating video: FPS={}".format(fps))
    clip = ImageSequenceClip(image_list, fps=fps)

    clip.write_videofile(video_file)


if __name__ == '__main__':
    main()
