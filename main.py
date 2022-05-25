import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn import metrics

df = pd.read_csv('/Users/xiechunyao/Downloads/tabular-playground-series-may-2022/train.csv')
df.set_index('id', inplace=True)
df.drop('f_27', axis=1, inplace=True)
targets = df['target'].to_numpy()
df.drop('target', axis=1, inplace=True)

train = df.to_numpy()[:200000]
val = df.to_numpy()[20001:]
targets_train = targets[:200000]
targets_val = targets[20001:]


# test_targets = test['target'].to_numpy()
# test.drop('target', axis=1, inplace=True)

clf = svm.SVC(kernel='linear')
clf.fit(train, targets_train)
test_pred = clf.predict(val)
print("Accuracy:",metrics.accuracy_score(targets_val, test_pred))
def show_train_scatter(targets,train):
    colors = []
    s = []
    train_1 = train['f_00'].to_numpy()
    train_2 = train['f_04'].to_numpy()
    for x in targets:
        s.append(2)
        if x == 1:
            colors.append('blue')
        else:
            colors.append('yellow')
    pca = PCA(n_components=2)
    compressed_embedding = pca.fit_transform(train)
    #plt.subplot(5, 4, (cnt - 1) * 4 + j + 1)
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(compressed_embedding[:,0],compressed_embedding[:,1],compressed_embedding[:,2],s=np.array(s),c=np.array(colors))
    #plt.scatter(compressed_embedding[:, 0],compressed_embedding[:, 1],s=np.array(s), c=np.array(colors))
    #plt.scatter(train_1,train_2, s=np.array(s),c=np.array(colors))
    plt.show()

