import pandas as pd
import numpy as np
  
import chainer
from chainer import serializers,Chain
import chainer.functions as F
import chainer.links as L
from chainer.backends import cuda 
 
 
#モデルの形を設定。こちらは、学習させた時と同じ形にする。
class MyChain(Chain):
 
    def __init__(self):
        super(MyChain, self).__init__(
        l1=L.Linear(4, 300).to_gpu(),
        l2=L.Linear(300, 200).to_gpu(),
        l3=L.Linear(200, 2).to_gpu(),      
    )
 
    def __call__(self, x):
        h1 = F.sigmoid(self.l1(x))
        h2 = F.sigmoid(self.l2(h1))
        o = self.l3(h2)
        return o
 
model = L.Classifier(MyChain()).to_gpu()
 
#学習済みモデルの読み込み
serializers.load_npz('C:\Python_Programs\chainer_practice\sampleNN.model', model)
 
#予測したいデータの読み込み
df = pd.read_csv("C:\Python_Programs\chainer_practice\sampleNN.csv")
N = len(df) #データの行数
 
#データの正規化。学習時におこなったものと同じものを行う。
df.iloc[:,:-1] /= df.iloc[:,:-1].max()
 
#入力データをnumpy配列に変更
np_data = np.array(df.iloc[:,:-1]).astype(np.float32)
data = cuda.to_gpu(np_data, device=0)
 
#予測後の出力ノードの配列を作成
outputArray = model.predictor(data).data
print(outputArray)
 
#予測結果の配列を作成
ansArray = np.argmax(outputArray,axis=1)
print(ansArray)
 
#出力ノードの値のデータフレーム版を作成
outputDF = pd.DataFrame(outputArray,columns=["output_0","output_1"])
 
#予測結果のデータフレーム版を作成
ansDF = pd.DataFrame(ansArray,columns=["PredictedValue"])
print(ansDF)
 
#真の値と、予測結果、出力ノードの値を格納したデータフレームを作成
result = pd.concat([df.disease,ansDF,outputDF],axis=1)
print(result)
 
#正解数、正答率を表示
correctCount = len(np.where(result.iloc[:,0] == result.iloc[:,1])[0])
correctRate = correctCount/N
print("データ数:",N)
print("正解数:",correctCount)
print("正答率:",correctRate)
 
#結果をcsvファイルへ出力
result.to_csv("samplePredict.csv",index=False)
