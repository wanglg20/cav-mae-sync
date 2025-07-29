import os
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torchvision.utils import save_image
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd
cv2.setNumThreads(0)

# preprocess pipeline
preprocess = T.Compose([
    T.Resize(224),
    T.CenterCrop(224),
    T.ToTensor()
])

def extract_frame(input_video_path: str, target_fold: str, extract_frame_num: int = 10, subfold: str = 'train'):
    """
    extract frames from video
    """
    target_fold = os.path.join(target_fold, subfold)
    if not os.path.exists(target_fold):
        os.makedirs(target_fold, exist_ok=True)
    ext = os.path.splitext(input_video_path)[1]
    video_id = os.path.basename(input_video_path).replace(ext, '')

    vidcap = cv2.VideoCapture(input_video_path)
    try:
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        total_frame_num = min(int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)), int(fps * 10))

        for i in range(extract_frame_num):
            frame_idx = int(i * (total_frame_num / extract_frame_num))
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, max(frame_idx - 1, 0))
            success, frame = vidcap.read()
            if not success:
                print(f'[{video_id}] Warning: failed to read frame {frame_idx}')
                continue

            # BGR → RGB → PIL → Tensor
            cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_im = Image.fromarray(cv2_im)
            image_tensor = preprocess(pil_im)
            
            frame_dir = os.path.join(target_fold, f'frame_{i}')
            if not os.path.exists(frame_dir):  # 避免每次都 makedirs
                os.makedirs(frame_dir, exist_ok=True)
            out_path = os.path.join(frame_dir, f'{video_id}.jpg')
            save_image(image_tensor, out_path)

            # 清理中间变量
            del cv2_im, pil_im, image_tensor, frame

    finally:
        vidcap.release()

def _worker(args):
    input_path, target_fold, num_frames, subfold = args
    try:
        extract_frame(input_path, target_fold, num_frames, subfold)
    except Exception as e:
        print(f'[ERROR] {input_path}: {e}')

if __name__ == "__main__":
    parser = ArgumentParser(description="Extract frames from videos")
    parser.add_argument(
        "-input_file_list", type=str,
        default='/home/chenyingying/tmp/cav-mae-sync/src/data_info/vggsound.csv',
        help="input file list"
    )
    parser.add_argument(
        "-target_fold", type=str,
        default='/data/wanglinge/dataset/VGGSound/frames_16',
        help="folder to save extracted frames"
    )
    parser.add_argument(
        "-num_workers", type=int, default=16,
        help="num of threads to use for parallel processing"
    )
    parser.add_argument(
        "-extract_frame_num", type=int, default=16,
        help="number of frames to extract from each video"
    )
    parser.add_argument(
        "-video_root", type=str, default='/data/wanglinge/dataset/VGGSound/scratch/shared/beegfs/hchen/train_data/VGGSound_final/video',
        help="root folder of videos"
    )
    args = parser.parse_args()
    if not os.path.exists(args.target_fold):
        os.makedirs(args.target_fold, exist_ok=True)
    input_filelist = pd.read_csv('/home/chenyingying/tmp/cav-mae-sync/src/data_info/vggsound.csv', 
                 names=['id', 'timestamp', 'label', 'split'],
                 header=None)
    input_filelist = [(os.path.join(args.video_root, f"{row['id']}_{str(row['timestamp']).zfill(6)}.mp4"), row['split']) for _, row in input_filelist.iterrows()]
    tasks = [(path, args.target_fold, args.extract_frame_num, subfold) for path, subfold in input_filelist]
    n_workers = args.num_workers

    batch_size = 500
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        for i in range(0, len(tasks), batch_size):
            batch_tasks = tasks[i:i+batch_size]
            futures = [executor.submit(_worker, task) for task in batch_tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing batch {i//batch_size+1}"):
                future.result()