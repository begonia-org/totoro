#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   media.py
@Time    :   2024/07/23 10:33:08
@Desc    :
'''
import tempfile
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import ffmpeg
import os
from funasr import AutoModel

from huggingface_hub import snapshot_download

from totoro.utils.utils import is_video, is_audio
from totoro.config import TotoroConfigure as cfg


class MediaParser:
    def __init__(self):
        # device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # # 模型ID
        # model_id = "openai/whisper-large-v3"
        # torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        # model_dir = self._download_model(model_id)
        # # 加载模型
        # model = AutoModelForSpeechSeq2Seq.from_pretrained(
        #     model_dir, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        # )
        # model.to(device)
        # processor = AutoProcessor.from_pretrained(model_id)
        # self.pipe = pipeline(
        #     "automatic-speech-recognition",
        #     model=model,
        #     tokenizer=processor.tokenizer,
        #     feature_extractor=processor.feature_extractor,
        #     chunk_length_s=25,  # 设置分块长度为25秒
        #     batch_size=16,  # 设置批处理大小
        #     torch_dtype=torch_dtype,
        #     device=device,
        # )
        self.model = AutoModel(
            model="/data2/work/models/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
            punc_model="/data2/work/models/punc_ct-transformer_cn-en-common-vocab471067-large",
            # trust_remote_code=True,
            # remote_code="./model.py",
            # vad_model="/data2/work/models/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
            # vad_kwargs={"max_single_segment_time": 5000},
            device="cpu",
            ncpu=8
        )
        pass

    def extract_audio_from_video(self, video_path, audio_path):
        ffmpeg.input(video_path).output(
            audio_path, format="wav").run(overwrite_output=True)

    def __call__(self, fnm):
        return self.apply(fnm)

    @staticmethod
    def row_number(fnm, binary):
        pass

    def _download_model(self, model_id: str):
        print(os.path.exists(os.path.join(
            cfg.model_dir(), os.path.basename(model_id))))
        if not os.path.exists(os.path.join(cfg.model_dir(), os.path.basename(model_id))):
            model_dir = snapshot_download(repo_id=model_id,
                                          local_dir=cfg.model_dir(),
                                          local_dir_use_symlinks=False)
            return model_dir

        return os.path.join(cfg.model_dir(), os.path.basename(model_id))

    def apply(self, fnm: str):
        if is_video(fnm):
            print(fnm)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as audio_file:
                self.extract_audio_from_video(fnm, audio_file.name)
                # text_file = f"{model.model_path}/example/text.txt"
                print(audio_file.name)
                res = self.model.generate(input=audio_file.name,
                                          language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
                                          cache={},
                                          #   use_itn=True,
                                          batch_size_s=32,
                                          #   merge_vad=True,  #
                                          #   merge_length_s=15
                                          )
                # return self.model(audio_file.name)
                return res
        # if is_audio(fnm):
        #     return self.pipe(fnm)
        return ""


if __name__ == "__main__":
    m = MediaParser()
    print(m("/data/work/begonia-org/openRAG/totoro/opendoc/parser/720.mp4"))
