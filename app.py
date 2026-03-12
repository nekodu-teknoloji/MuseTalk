import os
import time
import pdb
import re

import gradio as gr
import numpy as np
import sys
import subprocess

from huggingface_hub import snapshot_download
import requests

import argparse
import os
from omegaconf import OmegaConf
import numpy as np
import cv2
import torch
import glob
import pickle
from tqdm import tqdm
import copy
from argparse import Namespace
import shutil
import gdown
import imageio
import ffmpeg
from moviepy.editor import *
from transformers import WhisperModel

ProjectDir = os.path.abspath(os.path.dirname(__file__))
CheckpointsDir = os.path.join(ProjectDir, "models")

@torch.no_grad()
def debug_inpainting(video_path, bbox_shift, extra_margin=10, parsing_mode="jaw", 
                    left_cheek_width=90, right_cheek_width=90):
    if video_path is not None and hasattr(video_path, 'name'):
        video_path = video_path.name
    video_path = image_to_video(video_path)
    args_dict = {
        "result_dir": './results/debug', 
        "fps": 25, "batch_size": 1, "output_vid_name": '', 
        "use_saved_coord": False, "audio_padding_length_left": 2,
        "audio_padding_length_right": 2, "version": "v15",
        "extra_margin": extra_margin, "parsing_mode": parsing_mode,
        "left_cheek_width": left_cheek_width, "right_cheek_width": right_cheek_width
    }
    args = Namespace(**args_dict)
    os.makedirs(args.result_dir, exist_ok=True)
    
    if get_file_type(video_path) == "video":
        reader = imageio.get_reader(video_path)
        first_frame = reader.get_data(0)
        reader.close()
    else:
        first_frame = cv2.imread(video_path)
        first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    
    debug_frame_path = os.path.join(args.result_dir, "debug_frame.png")
    cv2.imwrite(debug_frame_path, cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR))
    
    coord_list, frame_list = get_landmark_and_bbox([debug_frame_path], bbox_shift)
    bbox = coord_list[0]
    frame = frame_list[0]
    
    if bbox == coord_placeholder:
        return None, "No face detected, please adjust bbox_shift parameter"
    
    fp = FaceParsing(left_cheek_width=args.left_cheek_width, right_cheek_width=args.right_cheek_width)
    
    x1, y1, x2, y2 = bbox
    y2 = y2 + args.extra_margin
    y2 = min(y2, frame.shape[0])
    crop_frame = frame[y1:y2, x1:x2]
    crop_frame = cv2.resize(crop_frame,(256,256),interpolation = cv2.INTER_LANCZOS4)
    
    random_audio = torch.randn(1, 50, 384, device=device, dtype=weight_dtype)
    audio_feature = pe(random_audio)
    latents = vae.get_latents_for_unet(crop_frame)
    latents = latents.to(dtype=weight_dtype)
    pred_latents = unet.model(latents, timesteps, encoder_hidden_states=audio_feature).sample
    recon = vae.decode_latents(pred_latents)
    
    res_frame = recon[0]
    res_frame = cv2.resize(res_frame.astype(np.uint8),(x2-x1,y2-y1))
    combine_frame = get_image(frame, res_frame, [x1, y1, x2, y2], mode=args.parsing_mode, fp=fp)
    
    debug_result_path = os.path.join(args.result_dir, "debug_result.png")
    cv2.imwrite(debug_result_path, combine_frame)
    
    info_text = f"Parameter information:\nbbox_shift: {bbox_shift}\nextra_margin: {extra_margin}\nparsing_mode: {parsing_mode}\nleft_cheek_width: {left_cheek_width}\nright_cheek_width: {right_cheek_width}\nDetected face coordinates: [{x1}, {y1}, {x2}, {y2}]"
    
    return cv2.cvtColor(combine_frame, cv2.COLOR_RGB2BGR), info_text


def download_model():
    required_models = {
        "MuseTalk": f"{CheckpointsDir}/musetalkV15/unet.pth",
        "SD VAE": f"{CheckpointsDir}/sd-vae/config.json",
        "Whisper": f"{CheckpointsDir}/whisper/config.json",
        "DWPose": f"{CheckpointsDir}/dwpose/dw-ll_ucoco_384.pth",
        "SyncNet": f"{CheckpointsDir}/syncnet/latentsync_syncnet.pt",
        "Face Parse": f"{CheckpointsDir}/face-parse-bisent/79999_iter.pth",
    }
    missing_models = [name for name, path in required_models.items() if not os.path.exists(path)]
    if missing_models:
        print("The following required model files are missing:")
        for model in missing_models:
            print(f"- {model}")
        print("\nPlease run the download script to download the missing models:")
        print("Linux/Mac: Run ./download_weights.sh")
        sys.exit(1)
    else:
        print("All required model files exist.")


download_model()

from musetalk.utils.blending import get_image
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.utils import get_file_type, get_video_fps, datagen, load_all_model
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder, get_bbox_range


def fast_check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except:
        return False


@torch.no_grad()
def inference(audio_path, video_path, bbox_shift, extra_margin=10, parsing_mode="jaw", 
              left_cheek_width=90, right_cheek_width=90, progress=gr.Progress(track_tqdm=True)):
    if video_path is not None and hasattr(video_path, 'name'):
        video_path = video_path.name
    video_path = image_to_video(video_path)
    args_dict = {
        "result_dir": './results/output', "fps": 25, "batch_size": 32,
        "output_vid_name": '', "use_saved_coord": False,
        "audio_padding_length_left": 2, "audio_padding_length_right": 2,
        "version": "v15", "extra_margin": extra_margin, "parsing_mode": parsing_mode,
        "left_cheek_width": left_cheek_width, "right_cheek_width": right_cheek_width
    }
    args = Namespace(**args_dict)

    if not fast_check_ffmpeg():
        print("Warning: Unable to find ffmpeg")

    input_basename = os.path.basename(video_path).split('.')[0]
    audio_basename = os.path.basename(audio_path).split('.')[0]
    output_basename = f"{input_basename}_{audio_basename}"
    
    temp_dir = os.path.join(args.result_dir, f"{args.version}")
    os.makedirs(temp_dir, exist_ok=True)
    
    result_img_save_path = os.path.join(temp_dir, output_basename)
    crop_coord_save_path = os.path.join(args.result_dir, "../", input_basename+".pkl")
    os.makedirs(result_img_save_path, exist_ok=True)

    output_vid_name = os.path.join(temp_dir, output_basename+".mp4") if args.output_vid_name == "" else os.path.join(temp_dir, args.output_vid_name)
        
    if get_file_type(video_path) == "video":
        save_dir_full = os.path.join(temp_dir, input_basename)
        os.makedirs(save_dir_full, exist_ok=True)
        reader = imageio.get_reader(video_path)
        for i, im in enumerate(reader):
            imageio.imwrite(f"{save_dir_full}/{i:08d}.png", im)
        input_img_list = sorted(glob.glob(os.path.join(save_dir_full, '*.[jpJP][pnPN]*[gG]')))
        fps = get_video_fps(video_path)
    else:
        input_img_list = sorted(glob.glob(os.path.join(video_path, '*.[jpJP][pnPN]*[gG]')), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        fps = args.fps
        
    whisper_input_features, librosa_length = audio_processor.get_audio_feature(audio_path)
    whisper_chunks = audio_processor.get_whisper_chunk(
        whisper_input_features, device, weight_dtype, whisper, librosa_length,
        fps=fps, audio_padding_length_left=args.audio_padding_length_left,
        audio_padding_length_right=args.audio_padding_length_right,
    )
        
    if os.path.exists(crop_coord_save_path) and args.use_saved_coord:
        with open(crop_coord_save_path,'rb') as f:
            coord_list = pickle.load(f)
        frame_list = read_imgs(input_img_list)
    else:
        print("extracting landmarks...time consuming")
        coord_list, frame_list = get_landmark_and_bbox(input_img_list, bbox_shift)
        with open(crop_coord_save_path, 'wb') as f:
            pickle.dump(coord_list, f)
    bbox_shift_text = get_bbox_range(input_img_list, bbox_shift)
    
    fp = FaceParsing(left_cheek_width=args.left_cheek_width, right_cheek_width=args.right_cheek_width)
    
    input_latent_list = []
    last_valid_bbox = None
    for bbox, frame in zip(coord_list, frame_list):
        if bbox == coord_placeholder:
            if last_valid_bbox is not None:
                bbox = last_valid_bbox
            else:
                h, w = frame.shape[:2]
                bbox = (w//4, h//4, 3*w//4, 3*h//4)
        last_valid_bbox = bbox
        x1, y1, x2, y2 = bbox
        y2 = min(y2 + args.extra_margin, frame.shape[0])
        crop_frame = cv2.resize(frame[y1:y2, x1:x2], (256,256), interpolation=cv2.INTER_LANCZOS4)
        input_latent_list.append(vae.get_latents_for_unet(crop_frame))

    frame_list_cycle = frame_list + frame_list[::-1]
    coord_list_cycle = coord_list + coord_list[::-1]
    input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
    
    print("start inference")
    video_num = len(whisper_chunks)
    batch_size = args.batch_size
    gen = datagen(whisper_chunks=whisper_chunks, vae_encode_latents=input_latent_list_cycle, batch_size=batch_size, delay_frame=0, device=device)
    res_frame_list = []
    for i, (whisper_batch, latent_batch) in enumerate(tqdm(gen, total=int(np.ceil(float(video_num)/batch_size)))):
        audio_feature_batch = pe(whisper_batch)
        latent_batch = latent_batch.to(dtype=weight_dtype)
        pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
        recon = vae.decode_latents(pred_latents)
        for res_frame in recon:
            res_frame_list.append(res_frame)
            
    print("pad talking image to original video")
    last_valid_bbox_cycle = None
    for i, res_frame in enumerate(tqdm(res_frame_list)):
        bbox = coord_list_cycle[i%(len(coord_list_cycle))]
        ori_frame = copy.deepcopy(frame_list_cycle[i%(len(frame_list_cycle))])
        if bbox == coord_placeholder:
            if last_valid_bbox_cycle is not None:
                bbox = last_valid_bbox_cycle
            else:
                h, w = ori_frame.shape[:2]
                bbox = (w//4, h//4, 3*w//4, 3*h//4)
        last_valid_bbox_cycle = bbox
        x1, y1, x2, y2 = bbox
        y2 = min(y2 + args.extra_margin, ori_frame.shape[0])
        try:
            res_frame = cv2.resize(res_frame.astype(np.uint8),(x2-x1,y2-y1))
        except:
            continue
        combine_frame = get_image(ori_frame, res_frame, [x1, y1, x2, y2], mode=args.parsing_mode, fp=fp)
        cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png", combine_frame)
        
    output_video = 'temp.mp4'
    images = []
    files = sorted([f for f in os.listdir(result_img_save_path) if re.compile(r'\d{8}\.png').match(f)], key=lambda x: int(x.split('.')[0]))
    for file in files:
        images.append(imageio.imread(os.path.join(result_img_save_path, file)))

    imageio.mimwrite(output_video, images, 'FFMPEG', fps=25, codec='libx264', quality=9, pixelformat='yuv420p')

    video_clip = VideoFileClip('./temp.mp4')
    audio_clip = AudioFileClip(audio_path)
    video_clip = video_clip.set_audio(audio_clip)
    video_clip.write_videofile(output_vid_name, codec='libx264', audio_codec='aac', fps=25)

    os.remove("temp.mp4")
    print(f"result is save to {output_vid_name}")
    return output_vid_name, bbox_shift_text


# load model weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae, unet, pe = load_all_model(
    unet_model_path="./models/musetalkV15/unet.pth", 
    vae_type="sd-vae",
    unet_config="./models/musetalkV15/musetalk.json",
    device=device
)

parser = argparse.ArgumentParser()
parser.add_argument("--ffmpeg_path", type=str, default=r"ffmpeg-master-latest-win64-gpl-shared\bin")
parser.add_argument("--ip", type=str, default="127.0.0.1")
parser.add_argument("--port", type=int, default=7860)
parser.add_argument("--share", action="store_true")
parser.add_argument("--use_float16", action="store_true")
args = parser.parse_args()

if args.use_float16:
    pe = pe.half()
    vae.vae = vae.vae.half()
    unet.model = unet.model.half()
    weight_dtype = torch.float16
else:
    weight_dtype = torch.float32

pe = pe.to(device)
vae.vae = vae.vae.to(device)
unet.model = unet.model.to(device)
timesteps = torch.tensor([0], device=device)

audio_processor = AudioProcessor(feature_extractor_path="./models/whisper")
whisper = WhisperModel.from_pretrained("./models/whisper")
whisper = whisper.to(device=device, dtype=weight_dtype).eval()
whisper.requires_grad_(False)


def image_to_video(image_path, duration=5, fps=25):
    if not isinstance(image_path, str):
        return image_path
    os.makedirs('./results/input', exist_ok=True)
    ext = os.path.splitext(image_path)[1].lower()
    if ext in ['.mp4', '.avi', '.mov', '.mkv']:
        return check_video(image_path)
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Could not read image: {image_path}")
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output_video = os.path.join('./results/input', os.path.splitext(os.path.basename(image_path))[0] + '_avatar.mp4')
    frames = [frame_rgb] * (duration * fps)
    imageio.mimwrite(output_video, frames, 'FFMPEG', fps=fps, codec='libx264', quality=9, pixelformat='yuv420p')
    print(f"Image converted to video: {output_video}")
    return output_video


def check_video(video):
    if not isinstance(video, str):
        return video
    dir_path, file_name = os.path.split(video)
    if file_name.startswith("outputxxx_"):
        return video
    output_file_name = "outputxxx_" + file_name
    os.makedirs('./results/input', exist_ok=True)
    output_video = os.path.join('./results/input', output_file_name)
    reader = imageio.get_reader(video)
    fps = reader.get_meta_data()['fps']
    frames = [im for im in reader]
    target_fps = 25
    L = len(frames)
    L_target = int(L / fps * target_fps)
    original_t = [x / fps for x in range(1, L+1)]
    t_idx = 0
    target_frames = []
    for target_t in range(1, L_target+1):
        while target_t / target_fps > original_t[t_idx]:
            t_idx += 1
            if t_idx >= L:
                break
        target_frames.append(frames[t_idx])
    imageio.mimwrite(output_video, target_frames, 'FFMPEG', fps=25, codec='libx264', quality=9, pixelformat='yuv420p')
    return output_video


css = """#input_img {max-width: 1024px !important} #output_vid {max-width: 1024px; max-height: 576px}"""

with gr.Blocks(css=css) as demo:
    gr.Markdown("""<div align='center'><h1>MuseTalk: Real-Time High-Fidelity Video Dubbing</h1></div>""")
    with gr.Row():
        with gr.Column():
            audio = gr.Audio(label="Driving Audio", type="filepath")
            video = gr.File(label="Reference Image or Video (jpg, png, mp4)", file_types=[".jpg", ".jpeg", ".png", ".mp4", ".avi", ".mov"])
            bbox_shift = gr.Number(label="BBox_shift value, px", value=0)
            extra_margin = gr.Slider(label="Extra Margin", minimum=0, maximum=40, value=10, step=1)
            parsing_mode = gr.Radio(label="Parsing Mode", choices=["jaw", "raw"], value="jaw")
            left_cheek_width = gr.Slider(label="Left Cheek Width", minimum=20, maximum=160, value=90, step=5)
            right_cheek_width = gr.Slider(label="Right Cheek Width", minimum=20, maximum=160, value=90, step=5)
            bbox_shift_scale = gr.Textbox(label="BBox Info")
            with gr.Row():
                debug_btn = gr.Button("1. Test Inpainting")
                btn = gr.Button("2. Generate")
        with gr.Column():
            debug_image = gr.Image(label="Test Inpainting Result (First Frame)")
            debug_info = gr.Textbox(label="Parameter Information", lines=5)
            out1 = gr.Video()
    
    def handle_file_upload(file):
        if file is None:
            return None
        path = file.name if hasattr(file, 'name') else file
        return image_to_video(path)

    video.change(fn=handle_file_upload, inputs=[video], outputs=[video])
    btn.click(fn=inference, inputs=[audio, video, bbox_shift, extra_margin, parsing_mode, left_cheek_width, right_cheek_width], outputs=[out1, bbox_shift_scale])
    debug_btn.click(fn=debug_inpainting, inputs=[video, bbox_shift, extra_margin, parsing_mode, left_cheek_width, right_cheek_width], outputs=[debug_image, debug_info])

if not fast_check_ffmpeg():
    print(f"Adding ffmpeg to PATH: {args.ffmpeg_path}")
    path_separator = ';' if sys.platform == 'win32' else ':'
    os.environ["PATH"] = f"{args.ffmpeg_path}{path_separator}{os.environ['PATH']}"
    if not fast_check_ffmpeg():
        print("Warning: Unable to find ffmpeg, please ensure ffmpeg is properly installed")

if sys.platform == 'win32':
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

demo.queue().launch(share=args.share, debug=True, server_name=args.ip, server_port=args.port)
