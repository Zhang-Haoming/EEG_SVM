import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.utils import check_random_state
from sklearn import svm, datasets
from sklearn.pipeline import Pipeline
import sklearn.model_selection as ms
from mne.decoding import CSP


def ShuffleSplit_SVM(trainData, trainLabels,kernel_type,Test_size):

    # 根据设计的交叉验证参数,分配相关的训练集和测试集数据   Assign training sets and test sets by cross validation parameter
    cv = ShuffleSplit(10, test_size=Test_size, random_state=42)

    # 创建SVM进行分类   create svm
    svc = SVC(kernel=kernel_type, class_weight='balanced')

    # 创建CSP提取特征，这里使用4个分量的CSP  create CSP to extract feature
    csp = CSP(n_components=4, reg=None, log=False, norm_trace=False)

    # 创建机器学习的Pipeline  create pipline, integration two method
    clf = Pipeline([('CSP', csp), ('svc', svc)])

    # 获取交叉验证模型的得分  get the score of cross validation parameter
    scores = cross_val_score(clf, trainData, trainLabels, cv=cv, n_jobs=-1)

    # 输出结果，准确率和不同样本的占比  get result , accuracy rate, chance level
    class_balance = np.mean(trainLabels == trainLabels[0])
    class_balance = max(class_balance, 1. - class_balance)

    return scores, class_balance


def GridSearch_SVM(trainData, trainLabels,param_grid):
    
    #分割数据集
    x_train, x_test, y_train, y_test = ms.train_test_split(trainData, trainLabels, random_state = 1, train_size = 0.8)
    #create and integrate SVM & CSP
    #svc = SVC(kernel='rbf', class_weight='balanced')
    svc = SVC( class_weight='balanced')
    csp = CSP(n_components=4, reg=None, log=False, norm_trace=False)
    clf = Pipeline([('CSP', csp), ('svc', svc)])

    #calling GridSearch
    grid = GridSearchCV(clf,param_grid,refit=True,cv=3,n_jobs=-1)    #n_jobs=-1：跟CPU核数一致 

    gs = grid.fit(x_train, y_train)
    train_grade = np.mean(gs.predict(x_train) == y_train)
    # 计算测试集精度 calculate the correct rate of test set, get the best parameter
    test_score = grid.score(x_test, y_test)
    para=grid.best_params_
    
    return train_grade, test_score, para