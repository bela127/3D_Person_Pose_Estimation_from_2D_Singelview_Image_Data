# -*- coding: utf-8 -*-
# ---------------------

# -*- coding: utf-8 -*-
from threading import Thread
import time
import json
import click
import sys
import imageio
import imageio_ffmpeg
from path import Path

assert sys.version_info >= (3, 6), '[!] This script requires Python >= 3.6'


def create_data(in_file_path, out_subdir_path, first_frame, img_format):
    video = in_file_path
    out_seq_path = out_subdir_path / video.basename().split('.')[0]
    if not out_seq_path.exists():
        out_seq_path.makedirs()
    reader = imageio.get_reader(video)
    print(f'▸ extracting frames of \'{Path(video).abspath()}\'')
    for frame_number, image in enumerate(reader):
        n = first_frame + frame_number
        imageio.imwrite(out_seq_path / f'{n}.{img_format}', image)
        print(f'\r▸ progress: {100 * (frame_number / 899):6.2f}%', end='')
    print()


class FrameDataCreatorThread(Thread):

    def __init__(self, in_file_path, out_subdir_path, first_frame, img_format):
        Thread.__init__(self)
        self.in_file_path = in_file_path
        self.out_subdir_path = out_subdir_path
        self.first_frame = first_frame
        self.img_format = img_format

    def run(self):
        print('[{}] > START'.format(self.in_file_path))
        create_data(self.in_file_path, self.out_subdir_path, self.first_frame, self.img_format)
        print('[{}] > DONE'.format(self.in_file_path))



def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


H1 = 'directory where you want to save the extracted frames'
H2 = 'number from which to start counting the video frames; DEFAULT = 1'
H3 = 'the format to use to save the images/frames; DEFAULT = png'
H4 = 'number of threads for multithreading'


# check python version
assert sys.version_info >= (3, 6), '[!] This script requires Python >= 3.6'


@click.command()
@click.option('--out_dir_path', type=click.Path(), default='./images', prompt='Enter \'out_dir_path\'', help=H1)
@click.option('--first_frame', type=int, default=1, help=H2)
@click.option('--img_format', type=str, default='png', help=H3)
@click.option('--n_threads', type=int, default=4, help=H4)
def main(out_dir_path, first_frame, img_format, n_threads):
    # type: (str, int, str) -> None
    """
    Script that splits all the videos into frames and saves them
    in a specified directory with the desired format
    """
    out_dir_path = Path(out_dir_path)
    if not out_dir_path.exists():
        out_dir_path.makedirs()

    for dir in Path('videos').dirs():
        out_subdir_path = out_dir_path / dir.basename()
        if not out_subdir_path.exists():
            out_subdir_path.makedirs()
        print(f'▸ extracting \'{dir.basename()}\' set')
        for video_chunk in chunks(dir.files(), n_threads):
            threads = []
            for video in video_chunk:
                threads.append(FrameDataCreatorThread(video, out_subdir_path, first_frame, img_format))

            for t in threads:
                t.start()

            for t in threads:
                t.join()



if __name__ == '__main__':
    main()