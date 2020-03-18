# coding:utf-8
import os
import glob
import random
import librosa
import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical


class SpecAugGenerator(Sequence):
    def __init__(self, dataset_path, cfg, seed=1024):
        self.cfg = cfg
        self.random_state = np.random.RandomState(seed)
        self._create_class_dict(dataset_path)
        self.queue = []
        self.index_per_class = [0] * self.class_num

    def __getitem__(self, idx):
        i = 0
        x, y = [], []
        while i < self.cfg['batch_size']:
            # class選択キューの補給
            if len(self.queue) == 0:
                self.queue = self._extend_queue(self.queue)

            # queueからクラス番号を抜き出し、ファイルパスを得る
            class_id = self.queue.pop(0)
            data_index = self.index_per_class[class_id]
            data_path = self.data_dict[class_id][data_index]

            # wavファイルを読み込み、メルスペクトログラムを作成
            signal, _ = librosa.core.load(data_path, self.cfg['sr'], mono=True)
            signal = self._clip_or_padding(signal)
            spec = librosa.feature.melspectrogram(
                signal, sr=self.cfg['sr'], n_mels=self.cfg['mel_bins'],
                n_fft=self.cfg['n_fft'], hop_length=self.cfg['hop_size']
            )
            spec_db = librosa.power_to_db(spec, ref=np.max)
            spec_db = spec_db - np.mean(spec_db)

            # spec_augmentを実施
            spec_masked = self._augment(spec_db)

            self.index_per_class[class_id] += 1
            if self.index_per_class[class_id] >= len(self.data_dict[class_id]):
                self.index_per_class[class_id] = 0
                self.random_state.shuffle(self.data_dict[class_id])
            x.append(np.expand_dims(spec_masked, axis=2))
            y.append(to_categorical(class_id, num_classes=self.class_num))
            i += 1
        return np.array(x), np.array(y)

    def _clip_or_padding(self, signal):
        # オーディオを指定の長さにクリッピング or パディングする
        limit_audio_samples = self.cfg['sr'] * self.cfg['audio_length']
        if signal.shape[0] < limit_audio_samples:
            signal = np.concatenate(
                (signal, np.zeros(limit_audio_samples - signal.shape[0])),
                axis=0
            )
        elif signal.shape[0] > limit_audio_samples:
            clipping_start_index = signal.shape[0]//2 - limit_audio_samples//2
            signal = signal[clipping_start_index:clipping_start_index+limit_audio_samples]
        return signal

    def _extend_queue(self, queue):
        # queueを補給する
        class_set = list(np.arange(self.class_num))
        self.random_state.shuffle(class_set)
        queue = class_set
        return queue

    def _augment(self, spec):
        # time warp is not implemented
        NFrame = spec.shape[1]
        NBin = spec.shape[0]

        # check
        if NFrame < self.cfg['spec_aug']['T_width'] * 2 + 1:
            T = NFrame//self.cfg['spec_aug']['T_line_num']
        else:
            T = self.cfg['spec_aug']['T_width']
        if NBin < self.cfg['spec_aug']['F_width'] * 2 + 1:
            F = NBin // self.cfg['spec_aug']['F_line_num']
        else:
            F = self.cfg['spec_aug']['F_width']
        t = np.random.randint(T-1, size=self.cfg['spec_aug']['T_line_num']) + 1
        f = np.random.randint(F-1, size=self.cfg['spec_aug']['F_line_num']) + 1
        mask_t = np.ones((NFrame, 1))
        mask_f = np.ones((NBin, 1))

        index = 0
        t_tmp = t.sum() + self.cfg['spec_aug']['T_line_num']
        for _t in t:
            k = random.randint(index, NFrame-t_tmp)
            mask_t[k:k+_t] = 0
            index += k + _t + 1
            t_tmp = t_tmp - (_t + 1)
        mask_t[index:] = 1
        index = 0
        f_tmp = f.sum() + self.cfg['spec_aug']['F_line_num']
        for _f in f:
            k = random.randint(index, NBin-f_tmp)
            mask_f[k:k+_f] = 0
            index += k + _f + 1
            f_tmp = f_tmp - (_f + 1)
        mask_f[index:] = 1

        spec_masked = ((spec * mask_t.T) * mask_f)
        return spec_masked

    def __len__(self):
        return self.data_num // self.cfg['batch_size']

    def on_epoch_end(self):
        for i in range(self.class_num):
            self.random_state.shuffle(self.data_dict[i])
        self.index_per_class = [0] * self.class_num
        return

    def _create_class_dict(self, dataset_path):
        self.data_dict = {}
        self.data_num = 0
        classes = glob.glob(os.path.join(dataset_path, "*"))
        self.class_num = len(classes)
        for i, cla in enumerate(classes):
            wav_files = glob.glob(os.path.join(cla, "*.wav"))
            self.data_num += len(wav_files)
            self.data_dict[i] = wav_files
