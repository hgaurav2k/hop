import os
from tkinter import commondialog
import numpy as np
import wandb
from PIL import Image 
import cv2 
class Logger:
    def __init__(self, log_dir, n_logged_samples=10, summary_writer=None):
        self._log_dir = log_dir
        print('########################')
        print('logging outputs to ', log_dir)
        print('########################')
        self._n_logged_samples = n_logged_samples
        self._summ_writer = summary_writer 

    def flush(self):
        self._summ_writer.flush()
        return 
    
    def log_scalar(self, scalar, name, step_, commit=False):
        if self._summ_writer:
            self._summ_writer.log({'{}'.format(name): scalar}, step=step_) #, commit=commit)

    def log_scalars(self, scalar_dict, group_name, step, phase, commit=True):
        """Will log all scalars in the same plot."""
        if self._summ_writer:
            self._summ_writer.log({'{}/{}'.format(group_name, phase): scalar_dict}, step=step) # Not sure if this will work!
        #self._summ_writer.add_scalars('{}_{}'.format(group_name, phase), scalar_dict, step)

    def log_image(self, image, name, step, commit=False):
        assert(len(image.shape) == 3)  # [C, H, W]
        image = wandb.Image(image, caption=f"{name}", step=step, commit=commit)
        #self._summ_writer.add_image('{}'.format(name), image, step)

    # TODO: Add more logging as needed
    def log_gifs(self,imgs,name="gif",commit=False):
        
        images = [Image.fromarray(image.cpu().numpy().astype(np.uint8)) for image in imgs]
        wandb.log({name: [wandb.Image(image) for image in images]})
    
    def log_video(self,imgs,name="video", step=0, commit=False, fps=15):
        
        frames = [img.cpu().numpy().astype(np.uint8) for img in imgs]
        frames = np.array(frames)  # [T, H, W, C]
        frames = np.transpose(frames, (0, 3, 1, 2))  # [T, C, H, W]        

        print("here")
        wandb.log({
                name: wandb.Video(frames, fps=fps, format='mp4'),
            }, step=step)
        
        print("here2")

    #def log_video(self, video_frames, name, step, fps=10):
    #    assert len(video_frames.shape) == 5, "Need [N, T, C, H, W] input tensor for video logging!"
    #    self._summ_writer.add_video('{}'.format(name), video_frames, step, fps=fps)

    #def log_trajs_as_videos(self, trajs, step, max_videos_to_save=2, fps=10, video_title='video'):

    #    # reshape the rollouts
    #    videos = [np.transpose(p['image_obs'], [0, 3, 1, 2]) for p in trajs]

    #    # max rollout length
    #    max_videos_to_save = np.min([max_videos_to_save, len(videos)])
    #    max_length = videos[0].shape[0]
    #    for i in range(max_videos_to_save):
    #        if videos[i].shape[0]>max_length:
    #            max_length = videos[i].shape[0]

    #    # pad rollouts to all be same length
    #    for i in range(max_videos_to_save):
    #        if videos[i].shape[0]<max_length:
    #            padding = np.tile([videos[i][-1]], (max_length-videos[i].shape[0],1,1,1))
    #            videos[i] = np.concatenate([videos[i], padding], 0)

    #    # log videos to tensorboard event file
    #    videos = np.stack(videos[:max_videos_to_save], 0)
    #    self.log_video(videos, video_title, step, fps=fps)

    #def log_figures(self, figure, name, step, phase):
    #    """figure: matplotlib.pyplot figure handle"""
    #    assert figure.shape[0] > 0, "Figure logging requires input shape [batch x figures]!"
    #    self._summ_writer.add_figure('{}_{}'.format(name, phase), figure, step)

    #def log_figure(self, figure, name, step, phase):
    #    """figure: matplotlib.pyplot figure handle"""
    #    self._summ_writer.add_figure('{}_{}'.format(name, phase), figure, step)

    #def log_graph(self, array, name, step, phase):
    #    """figure: matplotlib.pyplot figure handle"""
    #    im = plot_graph(array)
    #    self._summ_writer.add_image('{}_{}'.format(name, phase), im, step)

    #def dump_scalars(self, log_path=None):
    #    log_path = os.path.join(self._log_dir, "scalar_data.json") if log_path is None else log_path
    #    self._summ_writer.export_scalars_to_json(log_path)

    def log_dict(self, logs, itr, verbose=True):
        if self._summ_writer:
            for key, value in logs.items():
                if verbose:
                    print("{} : {}".format(key, value))
                self.log_scalar(value, key, itr)
