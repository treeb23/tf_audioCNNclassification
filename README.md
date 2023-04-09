# tf_audioCNNclassification

> ver 0.0.1 2023/04/09 仮

GoogleColaboratoryとVSCODE(Windows)で動作を確認

.ipynbファイルを以下の場所に作成する。各関数はディレクトリが存在しない場合に自動的に作成することはないため、必ずディレクトリを予め作成する。

ディレクトリ構成は
```
lab/
　├ code/
　│　├ ().ipynb
　│　└ save_model/
　│ 　 └ ().h5
　└ data/
　 　└ 短文音声/
    　 ├ test/
    　 │ └ thiswas/
    　 │    └ jp_1.wav
    　 ├ training/
    　 └ 画像/
        　├ test/
        　│ └ thiswas(mel)/
        　│   ├ jp/
        　│   │  └ jp_1.png
        　│   └ us/
        　└ training/
```

## tf_audioCNNclassificationの使い方
音声データを2種類用意して2クラス分類するための方法として画像に変換しCNNで学習,テストを行うサンプル。特徴量や変換のメカニズムを分からないで行なっているので精度は低い。

まず必要なライブラリのインストール
```py
!pip install librosa
```
インポート
```py
import matplotlib.pyplot as plt
import librosa
import librosa.display
import cv2
import keras
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import load_img, img_to_array
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.utils import np_utils
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import time
import datetime
from keras.models import model_from_json, load_model
```
このライブラリをインストールしてインポート
```py
!pip install git+https://github.com/treeb23/tf_audioCNNclassification.git
import tf_audioCNNclassification as tfacc
```

モデルの学習,テストの前には`set_param()`でパラメータを変更する。
```py
tfacc.set_param(imgname=['jp','us'],imgsize=200,color=1,epochnum=10,batchsize=32,featuren=0)
```

`imgname` : 学習/テストに用いる画像のファイル名2つ(['a','b']の形式)`{imgname}_1.png`

`imgsize` : 縦と横のピクセル数

`color` : 画像の色(0:グレースケール,1:カラー)

`epochnum` : エポック数

`batchsize` : バッチサイズ

`featuren` : 画像にするにあたって音声から取得する特徴量。

```
0: librosa.stft(wav)
1: librosa.feature.mfcc(y=wav,sr=sr,n_mfcc=20)
2: librosa.feature.melspectrogram(y=wav, sr=sr)
```


## trainingの方法

学習に使う音声データは次のパスにおく。

`lab/data/短文音声/training/{任意の名前}`

音声から変換された画像は次のパスにおく。

`lab/data/短文音声/画像/training/{任意の名前}`

音声のファイル名に振る連続する整数は0から始める。


## testの方法

テストに使う音声データは次のパスにおく。

`lab/data/短文音声/test/{任意の名前}`

音声から変換された画像は次のパスにおく。

`lab/data/短文音声/画像/test/{任意の名前}`

音声のファイル名に振る連続する整数は0から始める。


## その他の機能

### 音声から画像を用意する

`wav_name` : 学習に用いる音声のファイル名2つ(['a','b']の形式)`{wav_name}_1.wav`

`file_nums` : 音声/画像のファイル数2つ([10,20]の形式),この場合aが10,bが20個

```py
tfacc.create_img(wav_path="短文音声/test/thiswas",wav_name=['jpn','us'],img_path="短文音声/画像/training/thiswas(mel)",file_nums=[10,10])
```

### 学習する

```py
tfacc.training_CNN(model_name='cnn_model.h5',train_data_path="短文音声/画像/training/thiswas(mel)")
```

### テストする

モデルのネットワーク図を出力するには`view_model=True`とする(デフォルト)
```py
tfacc.pred(model_name='cnn_model.h5',test_data_path="短文音声/画像/training/thiswas(mel)",file_nums=[10,10],view_model=True)
```
