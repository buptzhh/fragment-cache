import numpy as np
from model import XGBoost
from metric import accuracy as acc
import time
import csv
# load data
# x_train_ = np.load('data/train_data.npy')
# y_train_ = np.load('data/train_target.npy')
# x_test = np.load('data/test_data.npy')
# y_test = np.load('data/test_target.npy')

# # split train data into train and eval data
# m, p = x_train_.shape
# np.random.seed(123)
# indices = np.arange(m)
# np.random.shuffle(indices)
# x_train = x_train_[indices[:round(0.8 * m)]]
# y_train = y_train_[indices[:round(0.8 * m)]]
# x_eval = x_train_[indices[round(0.8 * m):]]
# y_eval = y_train_[indices[round(0.8 * m):]]


# def default_params():
#     # create an xgboost model and fit it
#     xgb = XGBoost(
#         n_estimators=100,
#         random_state=123)
#     xgb.fit(x_train, y_train, eval_set=(x_eval, y_eval))

#     # predict and calculate acc
#     ypred_train = xgb.predict(x_train)
#     ypred_eval = xgb.predict(x_eval)
#     ypred_test = xgb.predict(x_test)
#     print("train acc = {0}".format(acc(y_train, ypred_train)))
#     print("eval acc = {0}".format(acc(y_eval, ypred_eval)))
#     print("test acc = {0}".format(acc(y_test, ypred_test)))

#     # plot learning curve to tune parameter
#     xgb.plot_learning_curve()


# def early_stop():
#     # create an xgboost model and fit it
#     xgb = XGBoost(
#         n_estimators=100,
#         random_state=123)
#     xgb.fit(x_train, y_train, eval_set=(x_eval, y_eval), early_stopping_rounds=20)
#     print('best iter: {}'.format(xgb.best_iter))

#     # predict and calculate acc
#     ypred_train = xgb.predict(x_train)
#     ypred_eval = xgb.predict(x_eval)
#     ypred_test = xgb.predict(x_test)
#     print("train acc = {0}".format(acc(y_train, ypred_train)))
#     print("eval acc = {0}".format(acc(y_eval, ypred_eval)))
#     print("test acc = {0}".format(acc(y_test, ypred_test)))

#     # plot learning curve to tune parameter
#     xgb.plot_learning_curve()


def tuned_params():
    # create an xgboost model and fit it
    x_train = []
    y_train = []
    fufile = open("../data/175_future.csv", "r")
    with open("../data/175_new.csv", "r") as infile:
        for line in infile:
            info = line.split(" ")
            ltime = int(info[0])
            delta = info[-3].split(",")[:-1]
            edc = info[-2].split(",")[:-1]
            # if ltime >=0:
            #     x_train.append(delta)
            #     x_train[-1] += edc
            
            line = fufile.readline()
            info = line.split(" ")
            ltime = int(info[0])
            future_time = int(info[3][:-1].split('.')[0])
            time_add = future_time - ltime
            if ltime >=0:
                # if future_time <0:
                #     time_add = 1200
                #     ccc+=1
                if future_time >=0:
                    y_train.append(time_add)
                    x_train.append(delta)
                    x_train[-1] += edc
            if ltime > 1200:
                break
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        print(x_train[-1])
        print(len(x_train))
    print("read finish")
    xgb = XGBoost(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        objective='reg:squarederror',
        gamma=0,
        reg_lambda=3,
        subsample=1,
        colsample=1,
        random_state=123)
    xgb.fit(x_train, y_train, eval_set=(x_train, y_train), early_stopping_rounds=20)
    print('best iter: {}'.format(xgb.best_iter))

    # predict and calculate acc
    ypred_train = xgb.predict(x_train)
    print(ypred_train)
    
    # ypred_eval = xgb.predict(x_eval)
    # ypred_test = xgb.predict(x_test)
    # print(x_test)
    # print(y_test)
    print("train acc = {0}".format(acc(y_train, ypred_train)))
    # print("eval acc = {0}".format(acc(y_eval, ypred_eval)))
    # print("test acc = {0}".format(acc(y_test, ypred_test)))
    x_test = []
    of = open("perdition.csv", "a+",newline="", encoding='UTF-8')
    writer = csv.writer(of, delimiter=' ')
    with open("data/175_new.csv", "r") as infile:
        for line in infile:
            info = line.split(" ")
            ltime = int(info[0])
            delta = info[-3].split(",")[:-1]
            edc = info[-2].split(",")[:-1]
            if ltime % 60 != 0:
                x_test.append(delta)
                x_test[-1] += edc
            else:
                y_test = xgb.predict(np.array(x_test))
                x_test.clear()
                x_test.append(delta)
                x_test[-1] += edc
                for i in y_test.tolist():
                    writer.writerow([i])
            if ltime>3600:
                break
    # plot learning curve to tune parameter
    # xgb.plot_learning_curve()


if __name__ == "__main__":
    s = time.time()

    # use default parameters without tuning
    # default_params()

    # use early stop
    # early_stop()
    # tuning parameters
    tuned_params()
    print(time.time() - s)
