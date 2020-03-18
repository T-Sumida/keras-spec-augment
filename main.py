# coding: utf-8
import yaml
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from spec_aug_generator import SpecAugGenerator

if __name__ == "__main__":
    # 初期設定
    dataset_path = "dataset/train"
    with open("config.yaml", 'r') as f:
        cfg = yaml.load(f)

    generator = SpecAugGenerator(dataset_path, cfg)
    X, Y = generator.__getitem__(0)
    # model.fit_generator(generator, step_per_epoch=generator.__len__(), epoch=10)

    # generatorが出力したスペクトログラムを確認
    fig = plt.figure()
    for i, (x, y) in enumerate(zip(X, Y)):
        ax = fig.add_subplot(cfg['batch_size'], 1, i+1)
        librosa.display.specshow(np.squeeze(x), y_axis='log', x_axis='time', sr=cfg['sr'], ax=ax)
    plt.subplots_adjust(top=0.9)
    plt.show()
