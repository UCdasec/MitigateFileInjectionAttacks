import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, VarianceThreshold
import glob,random


def read_csvfile(filename):
    X = pd.read_csv(filename)
    # X = X.drop('index',axis = 1)
    print("dataset's shape is {}".format(X.shape))
    return X

def preprocess_data(df):
    # drop 2 features manually based on observation
    X = df[['# of words','# of sentences','NN','VB','JJ','RB','PRP']]
    # X = df[['avg_stence_length','NN','VB','JJ','RB','PRP']]
    print("processed dataset's shape is {}".format(X.shape))
    return X

def label_creat(m,n):
    #m is # of negative label
    #n is # of positve label
     x1 = [0] * m
     x2 = [1] * n
     y = x1 + x2
     return y

def write_to_file(y_test,y_pred):
    col = ['y_test','y_predict']
    dic = {}
    dic[col[0]] = y_test
    dic[col[1]] = y_pred
    df = pd.DataFrame(dic)
    df = df[col]
    df.index = range(1,len(df) + 1)
    df.to_csv('result.csv')


def retrieve_file(path):
    filenames = glob.glob(path + '/*.csv')
    return filenames


def feature_select_chi2(X,y):
    # X is orginial dataset
    # y is label
    fit_func = SelectKBest(score_func=chi2,k=7).fit(X,y)
    print ('*'*30)
    X_new = fit_func.transform(X)
    print ("After feature selection, dataset's shape is {}".format(X_new.shape))
    print(X_new)
    return X_new


def feture_select_variance(X):
    # remove the fetures with low variance based on threhold
    variance_func = VarianceThreshold(30)
    X_new = variance_func.fit_transform(X)
    print ('-'*70)
    print ('======= dataset after variance_selection ========')
    print (X_new)
    return X_new


def model_select(X,y):
    svm_class = SVC()
    rf_class = RandomForestClassifier(n_estimators = 10)
    log_class = LogisticRegression()
    abc_default = AdaBoostClassifier(n_estimators=50,learning_rate=1)   #abs use decision tree as default weak base learner
    abc_rf = AdaBoostClassifier(n_estimators=50,base_estimator=rf_class,learning_rate=1)
    abc_svm = AdaBoostClassifier(n_estimators=50,base_estimator=SVC(probability=True, kernel='linear'),learning_rate=1)
    models = [svm_class,rf_class,log_class,abc_default,abc_rf,abc_svm]
    model_name = ['SVM','RandomFrest','LogisticRegression','abc_default','abc_rf','abc_svm']
    for i in range(len(models)):
        score_model = cross_val_score(models[i],X,y,scoring = 'accuracy', cv=10).mean()
        print 'average accuracy of {} model is {}'.format(model_name[i],score_model)
        print '\n'


def selected_classifiers(X,y):
    x_train,x_test,y_train,y_test = train_test_split(X,y, test_size = 0.2)
    rf_class = RandomForestClassifier(n_estimators = 10)
    abc_rf = AdaBoostClassifier(n_estimators=50, base_estimator=rf_class, learning_rate=1)
    abc_default = AdaBoostClassifier(n_estimators=50, learning_rate=1)
    models = [rf_class,abc_rf,abc_default]
    models_name = ['rf','abc_rf','abc_default']
    i=0
    for model in models:
        model.fit(x_train,y_train)
        y_pred = model.predict(x_test)
        print '\n', '{} model '.format(models_name[i])
        print classification_report(y_test,y_pred)
        print 'confusion matrix is {}'.format(confusion_matrix(y_test,y_pred))
        print 'accuracy is {}'.format(metrics.accuracy_score(y_test,y_pred))
        i += 1


def cross_classification(x_train,y_train,x_test,y_test):
    "either use ngram_dataset or rnn_dataset for training, then use the rest one as test dataset"

    rf_class = RandomForestClassifier(n_estimators=10)
    abc_rf = AdaBoostClassifier(n_estimators=50, base_estimator=rf_class, learning_rate=1)
    abc_default = AdaBoostClassifier(n_estimators=50, learning_rate=1)
    models = [rf_class, abc_rf, abc_default]
    models_name = ['rf', 'abc_rf', 'abc_default']
    i = 0
    for model in models:
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print '\n', '{} model '.format(models_name[i])
        print classification_report(y_test, y_pred)
        print 'confusion matrix is {}'.format(confusion_matrix(y_test, y_pred))
        print 'accuracy is {}'.format(metrics.accuracy_score(y_test, y_pred))
        # print y_test
        # print y_pred
        i += 1


def twoModel_cross_classification(training_filepath, testing_filepath):
    # rnn_dataset and ngram_dataset
    # one for training and one for testing

    def build_dataset(trainset_file,textset_file):
        df_train = read_csvfile(trainset_file)
        x_train = preprocess_data(df_train)
        y_train = df_train[['label']]
        df_test = read_csvfile(textset_file)
        x_test = preprocess_data(df_test)
        y_test = df_test[['label']]
        return x_train,y_train,x_test,y_test

    x_train, y_train, x_test, y_test = build_dataset(training_filepath, testing_filepath)
    cross_classification(x_train,y_train,x_test,y_test)


def main_twoModel(ngram_feature_path,rnn_feature_path):
    " case1: training data: ngram, testing data: rnn; "
    " case2: training data: rnn, testing data: ngram "

#     ngram_feature_path = '../result/features/ngram_rc_generator/num1000_len50_inject20/append'
#     rnn_feature_path = '../result/features/rnn_rc_epoch100/num1000_len50/append'
    ngram_feature_names = retrieve_file(ngram_feature_path)
    rnn_feature_names = retrieve_file(rnn_feature_path)
    # CASE1: training_data: ngram; testing_data: rnn
    training_filepath = random.sample(ngram_feature_names, 1)
    i = 0
    for testing_filepath in rnn_feature_names:
        i += 1
        print('case {}...testing dataset is {}...'.format(i,testing_filepath))
        twoModel_cross_classification(training_filepath[0],testing_filepath)
    print('*' * 30)

    # CASE2: training_data: rnn; testing_data: ngram
    training_filepath = random.sample(rnn_feature_names,1)
    j = 0
    for testing_filepath in ngram_feature_names:
        j += 1
        print('case {}...testing dataset is {}...'.format(j,testing_filepath))
        twoModel_cross_classification(training_filepath[0],testing_filepath)


def main_chi(filename):
    df = read_csvfile(filename)
    X = df[['# of words','# of sentences','avg_stence_length','avg_word_length','NN','VB','JJ','RB','PRP']]
    y = df[['label']]
    print(X.head())
    print type(X)
    print type(y)
    X_new = feature_select_chi2(X,y)



#     test(X_new,y)


def main(filename):

    df = read_csvfile(filename)

    # 2 features removed based on obeservation
    X = preprocess_data(df)
    y = df[['label']]

    print(X.head())
    selected_classifiers(X, y)


    # without selecting features, to test all classifiers on all 9 features
    #model_select(X,y)

    # with selection, test all classifiers on selected features
    # X = preprocess_data(X)
    # model_select(X,y)

    #all 9 features
    # print X.head()
    # test(X,y)

    # X_new = feature_select_chi2(X,y)
    # print '='*15
    # test(X_new,y)

    # X_new = feture_select_variance(X)
    # test(X_new,y)


def main_forLoop(main_symbol,filenames):
    # main_symbol: True, choose main() function;
    # main_symbol: False, choose twoModel_cross_classification function


    for filename in filenames:
        if main_symbol == True:
            print('loop {}...'.format(filename))
            main(filename)
            print('\n')
        else:
            print('loop {}...'.format(filename))
            # ngram_filepath = '../result/features/ngram_rc_generator/num250/inserted_keywords20/feature_append.csv'
            # rnn_filepath = '../result/features/rnn_rc_epoch100/num250_len50/feature_append.csv'
            pass



if __name__ == '__main__':

    # filepath = '../result/features/ngram_rc_generator/keyword_science_remove_stopword/num317_len160_inject80/append'
    # filepath = '../result/features/rnn_rc_epoch200/keyword_science_remove_stopword/num316_len160_inject80/append'
    # filepath = '../result/features/rnn_raw_text_epoch100/num316_len160_inject80/append'
    # filepath = '../result/features/amazon_review/science/rnn/epoch40/inject40/append'
    filepath = '../result/features/amazon_review/science/ngram/inject40/append'

    "feature selection"
    # main_chi(filepath)



    " muti-running / single running"
    filenames = retrieve_file(filepath)
    main_symbol = True
    main_forLoop(main_symbol,filenames)


    " single running "
    # main(filenames)


    " two model cross classification "
    # ngram_feature_path = '../result/features/ngram_rc_generator/keyword_science_remove_stopword/num317_len160_inject80/append'
    # rnn_feature_path = '../result/features/rnn_rc_epoch100/keyword_science_remove_stopword/num316_len160_inject80/append'

    # ngram_feature_path = '../result/features/amazon_review/enron_keywords/ngram/inject40/append'
    # rnn_feature_path = '../result/features/amazon_review/enron_keywords/rnn/epoch60/inject40/append'
    # main_twoModel(ngram_feature_path,rnn_feature_path)
