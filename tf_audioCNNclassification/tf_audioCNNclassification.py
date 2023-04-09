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


def set_param(imgname=['jp','us'],imgsize=200,color=1,epochnum=10,batchsize=32,featuren=0):
    global f_path,folder,image_size,color_setting,epoch,batch,feature
    f_path=".."
    try:
        # Google Driveをcolabにマウント
        from google.colab import drive
        drive.mount('/content/drive')
        f_path="/content/drive/MyDrive/lab"
    except ModuleNotFoundError as e:
        print(e)
    print(f"[ファイルパスf_pathを'{f_path}'に設定]")
    folder=imgname #必ず半角英数
    image_size=imgsize
    if color==0:
        color_setting=1 #グレースケール
    else:
        color_setting=3 #カラー
    epoch=epochnum
    batch=batchsize
    feature=featuren


# 画像作成
def create_img(wav_path="短文音声/test/thiswas",wav_name=['jpn','us'],img_path="短文音声/画像/training/thiswas(mel)",file_nums=[10,10]):
    for y in range(2):
        for i in range(file_nums[y]):
            audio_path = f"{f_path}/data/{wav_path}/{wav_name[y]}_{i}.wav"
            wav,sr=librosa.load(audio_path,sr=16000)
            if feature==0:
                D = librosa.stft(wav) # 特徴量抽出関数
            elif feature==1:
                D = librosa.feature.mfcc(y=wav,sr=sr,n_mfcc=20)
                # y->波形データ, sr ->サンプリングレート, n_mfcc->最後の低次元抽出で何次元とるか
            elif feature==2:
                D = librosa.feature.melspectrogram(y=wav, sr=sr)
            S, phase = librosa.magphase(D)  # 複素数を強度と位相へ変換
            Sdb = librosa.amplitude_to_db(S)  # 強度をdb単位へ変換
            plt.figure(figsize=(2, 2), dpi=200)
            librosa.display.specshow(Sdb, sr=sr, x_axis='time', y_axis='log')  # スペクトログラムを表示
            plt.subplots_adjust(left=0, right=1, bottom=0, top=1) #余白を調整
            plt.savefig(f"{f_path}/data/{img_path}/{folder[y]}/{folder[y]}_{i}.png")
            plt.close()

    im = cv2.imread(f"{f_path}/data/{img_path}/{folder[y]}/{folder[0]}_{0}.png")
    print("高さ",im.shape[0]," px, 幅",im.shape[1]," px")


# データセットの読み込みとデータ形式の設定・正規化・分割、畳み込みニューラルネットワーク（CNN）・学習の実行等
def training_CNN(model_name='cnn_model.h5',train_data_path="短文音声/画像/training/thiswas(mel)"):
    class_number = len(folder)
    X_image = []
    Y_label = []
    for index, name in enumerate(folder):
        read_data = f"{f_path}/data/{train_data_path}/{name}"
        files = glob.glob(read_data + '/*.png')
        print('--- 読み込んだデータセット:', read_data)

        for i, file in enumerate(files):
            if color_setting == 1:
                img = load_img(file, color_mode = 'grayscale' ,target_size=(image_size, image_size))
            elif color_setting == 3:
                img = load_img(file, color_mode = 'rgb' ,target_size=(image_size, image_size))
            array = img_to_array(img)
            X_image.append(array)
            Y_label.append(index)

    X_image = np.array(X_image)
    Y_label = np.array(Y_label)

    X_image = X_image.astype('float32') / 255
    #Y_label = keras.utils.to_categorical(Y_label, class_number) #Kerasのバージョンなどにより使えないのでコメントアウト
    Y_label = np_utils.to_categorical(Y_label, class_number) #上記のコードのかわり

    train_images, valid_images, train_labels, valid_labels = train_test_split(X_image, Y_label, test_size=0.10)
    x_train = train_images
    y_train = train_labels
    x_test = valid_images
    y_test = valid_labels

    model = Sequential()
    model.add(Conv2D(16, (3, 3), padding='same',
            input_shape=(image_size, image_size, color_setting), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(class_number, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                optimizer=Adam(),
                metrics=['accuracy'])

    start_time = time.time()

    history = model.fit(x_train,y_train, batch_size=batch, epochs=epoch, verbose=0, validation_data=(x_test, y_test))
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(f'{f_path}/data/{train_data_path}/rep_{feature}.csv')

    model.save(model_name)

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Loss:', score[0], '（損失関数値 - 0に近いほど正解に近い）')
    print('Accuracy:', score[1] * 100, '%', '（精度 - 100% に近いほど正解に近い）')
    print('Computation time:{0:.3f} sec（秒）'.format(time.time() - start_time))
    with open(f'{f_path}/data/{train_data_path}/rep_{feature}.txt', 'a') as f:
        print('学習日時 : ',datetime.datetime.now(), file=f)
        print('Loss:', score[0], '（損失関数値 - 0に近いほど正解に近い）', file=f)
        print('Accuracy:', score[1] * 100, '%', '（精度 - 100% に近いほど正解に近い）', file=f)
        print('Computation time（計算時間）:{0:.3f} sec（秒）'.format(time.time() - start_time), file=f)


# 予測と結果
def pred(model_name='cnn_model.h5',test_data_path="短文音声/画像/training/thiswas(mel)",file_nums=[10,10],view_model=True):
    model = load_model(model_name)
    y=0
    exports=0
    export= [[0 for j in range(2)] for i in range(max(file_nums))]
    for t in folder:
        for j in range(file_nums[y]):
            recognise_image = f'{f_path}/data/{test_data_path}/{folder[y]}/{folder[y]}_{j}.png'
            img = cv2.imread(recognise_image, 1)
            img = cv2.resize(img, (image_size, image_size))
            img = img.reshape(image_size, image_size, color_setting).astype('float32')/255
            prediction = model.predict(np.array([img]), batch_size=batch, verbose=0)
            result = prediction[0]
            for i, accuracy in enumerate(result):
                export[j][y]=int(accuracy*100)
                #print('「', folder[i], '」の確率を', int(accuracy * 100), '% と予測しました。')
            if folder[y]==folder[result.argmax()]:
                exports=exports+1
            y=y+1
    exports=exports/(file_nums[0]+file_nums[1])
    print("抽出する特徴量 : ",feature)
    print("epoch=",epoch,", batch=",batch,"image_size=",image_size)
    print("正確に判断された率",exports*100," [%]\n")
    with open(f'{f_path}/data/{test_data_path}/rep_{feature}.txt', 'a') as fp:
        print('テスト日時 : ',datetime.datetime.now(), file=fp)
        print('v-------------------------------------------------------',file=fp)
        model.summary(print_fn=lambda x: fp.write(x + "\n"))
        print("抽出する特徴量 : ",feature,"\n","epoch=",epoch,", batch=",batch," ,image_size=",image_size,"\n","正確に判断された率",exports*100," [%]\n", file=fp)
        print('^-------------------------------------------------------',"\n",file=fp)
    # モデルの可視化 (samples, height, width, channels)
    # https://qiita.com/ak11/items/67118e11b756b0ee83a5
    if view_model==True:
        keras.utils.plot_model(model, to_file=f'{f_path}/data/{test_data_path}/model.png', show_shapes=True)
    return export
