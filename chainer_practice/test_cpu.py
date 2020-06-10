import pandas as pd
import numpy as np
  
import chainer
from chainer import serializers,Chain
import chainer.functions as F
import chainer.links as L
 
 
#モデルの形を設定。こちらは、学習させた時と同じ形にする。
class MyChain(Chain):
 
    def __init__(self):
        super(MyChain, self).__init__(
        l1=L.Linear(4, 800),
        l2=L.Linear(800, 400),
        l3=L.Linear(400, 2),      
    )
 
    def __call__(self, x):
        h1 = F.sigmoid(self.l1(x))
        h2 = F.sigmoid(self.l2(h1))
        o = self.l3(h2)
        return o
 
model = L.Classifier(MyChain())
 
#学習済みモデルの読み込み
serializers.load_npz('sampleNN.model', model)
 
#予測したいデータの読み込み
df = pd.read_csv("sampleNN.csv")
N = len(df) #データの行数
 
#データの正規化。学習時におこなったものと同じものを行う。
df.iloc[:,:-1] /= df.iloc[:,:-1].max()
 
#入力データをnumpy配列に変更
data = np.array(df.iloc[:,:-1]).astype(np.float32)
 
#予測後の出力ノードの配列を作成
outputArray = model.predictor(data).data
 
#予測結果の配列を作成
ansArray = np.argmax(outputArray,axis=1)
 
#出力ノードの値のデータフレーム版を作成
outputDF = pd.DataFrame(outputArray,columns=["output_0","output_1"])
 
#予測結果のデータフレーム版を作成
ansDF = pd.DataFrame(ansArray,columns=["PredictedValue"])
 
#真の値と、予測結果、出力ノードの値を格納したデータフレームを作成
result = pd.concat([df.disease,ansDF,outputDF],axis=1)
 
#正解数、正答率を表示
correctCount = len(np.where(result.iloc[:,0] == result.iloc[:,1])[0])
correctRate = correctCount/N
print("データ数:",N)
print("正解数:",correctCount)
print("正答率:",correctRate)
 
#結果をcsvファイルへ出力
result.to_csv("samplePredict.csv",index=False)
