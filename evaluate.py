import os
import deepspeech
import librosa
import soundfile as sf
import numpy as np


def main(wav_dir, txt_dir, model_params):
  print("Starting evaluation")
  wav_files = os.listdir(wav_dir)

  deepspeech_model = deepspeech.Model(model_params['model'], model_params['beam_width'])
  deepspeech_model.enableDecoderWithLM(model_params['lm'], model_params['trie'], 0.75, 1.85)
  print(deepspeech_model.sampleRate())

  for wav_file in wav_files:
    print('\n', wav_file)
    audio_array, samplerate = sf.read(os.path.join(wav_dir, wav_file))
    audio_arr, sr = librosa.core.load(os.path.join(wav_dir, wav_file), sr=16000)

    audio_arr = (audio_arr * 32767).astype(np.int16)
    words = deepspeech_model.stt(audio_arr)
    print(words)

if __name__ == '__main__':
    model_params = {
      "model": 'data/models/deepspeech-0.6.1-models/output_graph.pbmm',
      "lm":'data/models/deepspeech-0.6.1-models/lm.binary',
      "trie":'/data/models/deepspeech-0.6.1-models/trie',
      "beam_width":500
    }
    main(wav_dir='data/wav', txt_dir='data/txt', model_params=model_params)
