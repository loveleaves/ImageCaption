import os

os.system('git clone https://github.com/pytorch/fairseq.git; cd fairseq;'
          'pip install --use-feature=in-tree-build ./; cd ..')
os.system('curl -L ip.tool.lu; pip install torchvision')
os.system('bash caption_large_best.sh; mkdir -p checkpoints; mv caption_large_best.pt checkpoints/caption.pt')

import torch
import numpy as np
import re
from fairseq import utils,tasks
from fairseq import checkpoint_utils
from utils.eval_utils import eval_step
from tasks.mm_tasks.caption import CaptionTask
from models.ofa import OFAModel
from PIL import Image
from torchvision import transforms
import gradio as gr
import sys
import uuid
import requests
import wave
import base64
import hashlib
from imp import reload
import time

reload(sys)

APP_KEY = '0f3e5006d4c9e72a'
APP_SECRET = 'H7zqPQyJlTOPxVmfvFVeMNcolKxQXREF'

# Register caption task
tasks.register_task('caption', CaptionTask)
# turn on cuda if GPU is available
use_cuda = torch.cuda.is_available()
# use fp16 only when GPU is available
use_fp16 = False

# Load pretrained ckpt & config
overrides = {"bpe_dir": "utils/BPE", "eval_cider": False, "beam": 5,
             "max_len_b": 16, "no_repeat_ngram_size": 3, "seed": 7}
models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
    utils.split_paths('checkpoints/caption.pt'),
    arg_overrides=overrides
)

# Move models to GPU
for model in models:
    model.eval()
    if use_fp16:
        model.half()
    if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
        model.cuda()
    model.prepare_for_inference_(cfg)

# Initialize generator
generator = task.build_generator(models, cfg.generation)

mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

patch_resize_transform = transforms.Compose([
    lambda image: image.convert("RGB"),
    transforms.Resize((cfg.task.patch_image_size, cfg.task.patch_image_size), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

# Text preprocess
bos_item = torch.LongTensor([task.src_dict.bos()])
eos_item = torch.LongTensor([task.src_dict.eos()])
pad_idx = task.src_dict.pad()


def encode_text(text, length=None, append_bos=False, append_eos=False):
    s = task.tgt_dict.encode_line(
        line=task.bpe.encode(text),
        add_if_not_exist=False,
        append_eos=False
    ).long()
    if length is not None:
        s = s[:length]
    if append_bos:
        s = torch.cat([bos_item, s])
    if append_eos:
        s = torch.cat([s, eos_item])
    return s


# Construct input for caption task
def construct_sample(image: Image):
    patch_image = patch_resize_transform(image).unsqueeze(0)
    patch_mask = torch.tensor([True])
    src_text = encode_text(" what does the image describe?", append_bos=True, append_eos=True).unsqueeze(0)
    src_length = torch.LongTensor([s.ne(pad_idx).long().sum() for s in src_text])
    sample = {
        "id": np.array(['42']),
        "net_input": {
            "src_tokens": src_text,
            "src_lengths": src_length,
            "patch_images": patch_image,
            "patch_masks": patch_mask
        }
    }
    return sample


# Function to turn FP32 to FP16
def apply_half(t):
    if t.dtype is torch.float32:
        return t.to(dtype=torch.half)
    return t

def truncate(q):
    if q is None:
        return None
    size = len(q)
    return q if size <= 20 else q[0:10] + str(size) + q[size-10:size]

def encrypt(signStr):
    hash_algorithm = hashlib.sha256()
    hash_algorithm.update(signStr.encode('utf-8'))
    return hash_algorithm.hexdigest()

# sentence translation
def sentence_trans(q,Trans_to = "auto"):
    data = {}
    data['from'] = 'auto'
    data['to'] = Trans_to
    data['signType'] = 'v3'
    curtime = str(int(time.time()))
    data['curtime'] = curtime
    salt = str(uuid.uuid1())
    signStr = APP_KEY + truncate(q) + salt + curtime + APP_SECRET
    sign = encrypt(signStr)
    data['appKey'] = APP_KEY
    data['q'] = q
    data['salt'] = salt
    data['sign'] = sign
    data['vocabId'] = "2D4552567D81424D91FBF4805C70E05A"

    # 数据请求
    YOUDAO_SENTENCE_URL = 'https://openapi.youdao.com/api'
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    response = requests.post(YOUDAO_SENTENCE_URL, data=data, headers=headers)
    contentType = response.headers['Content-Type']
    content = str(response.content,'utf-8')
    # print(content)
    false = 0
    true = 1 # 用于处理str转dict时key值为true值未定义的情况
    content = eval(content)
    answer = ""
    try:
        answer = content["translation"][0]
    except Exception as e:
        answer = content["web"][0]["value"][0]
    # print(answer)
    return answer

# audio generate
def audio_generate_encrypt(signStr):
    hash_algorithm = hashlib.md5()
    hash_algorithm.update(signStr.encode('utf-8'))
    return hash_algorithm.hexdigest()

def audio_generate(q):
    data = {}
    data['langType'] = 'zh-CHS'
    salt = str(uuid.uuid1())
    signStr = APP_KEY + q + salt + APP_SECRET
    sign = audio_generate_encrypt(signStr)
    data['appKey'] = APP_KEY
    data['q'] = q
    data['salt'] = salt
    data['sign'] = sign

    # 数据请求
    YOUDAO_AUDIO_GENERATE_URL = 'https://openapi.youdao.com/ttsapi'
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    response = requests.post(YOUDAO_AUDIO_GENERATE_URL, data=data, headers=headers)
    contentType = response.headers['Content-Type']
    # millis = int(round(time.time() * 1000))
    # filePath = "audio" + str(millis) + ".mp3"
    filePath = "audio_answer.mp3"
    if os.path.isfile(filePath):
        os.system("rm -rf " + filePath)
    fo = open(filePath, 'wb')
    fo.write(response.content)
    fo.close()
    # audio_answer = AudioSegment.from_file("audio_answer.mp3", format = 'MP3')
    # os.system("ffmpeg -i 'temp_audio.wav' -ar 16000 'temp_audio_new.wav'")
    # audio_answer = list(audio_answer._data)
    # audio_answer = np.array(audio_answer)
    # print(audio_answer)
    # return (48000,audio_answer)
    return filePath

# Function for image captioning
def image_caption(Image):
    sample = construct_sample(Image)
    sample = utils.move_to_cuda(sample) if use_cuda else sample
    sample = utils.apply_to_sample(apply_half, sample) if use_fp16 else sample
    with torch.no_grad():
        result, scores = eval_step(task, generator, models, sample)
    answer = result[0]['caption']
    sentence_answer = sentence_trans(answer, "zh-CHS")
    audio_answer = audio_generate(sentence_answer)
    return (sentence_answer,audio_answer)


title = "Image_Caption"
description = "盲人避障系统的Demo。 食用指南：上传一张图片或点击一张示例图片, 然后点击 " \
              "\"Submit\" 等待些许即可得到 VFB's 回答结果。 "
article = "<p style='text-align: center'><a href='https://github.com/loveleaves/ImageCaption' target='_blank'>VFB Github " \
          "Repo</a></p> "
examples = [['example-1.jpg'], ['example-2.jpg'], ['example-3.jpg'], ['example-4.jpg'], ['example-5.jpg']]
io = gr.Interface(fn=image_caption, inputs=gr.inputs.Image(type='pil'), outputs=[gr.outputs.Textbox(label="Caption"),gr.outputs.Audio(type="file")],
                  title=title, description=description, article=article, examples=examples,
                  allow_flagging=False, allow_screenshot=False)
io.launch(cache_examples=True)
