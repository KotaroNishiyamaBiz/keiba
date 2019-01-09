import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def create_is_win(order_of_finish):
    if order_of_finish == 1:
        return 1
    else:
        return 0


def create_vote_rate(odds):
    if odds == 0:
        return 0
    else:
        return 1/odds


df = pd.read_csv('keiba_data.csv')
df["is_win"] = df.order_of_finish.apply(create_is_win)
df["rate"] = df.odds.apply(create_vote_rate)


y_datas = pd.DataFrame(df['is_win'], columns=['is_win']).values
x_datas = pd.DataFrame(df['rate'], columns=['rate']).fillna(0).values
odds_datas = pd.DataFrame(df['odds'], columns=['odds']).values


n_train_rate = 0.7
train_x_data, test_x_data, train_y_data, test_y_data, train_odds_data, test_odds_data = train_test_split(x_datas, y_datas, odds_datas, test_size=1-n_train_rate)

lr = LogisticRegression()
lr.fit(train_x_data, train_y_data)

a = lr.coef_
b = lr.intercept_

predict_df = pd.concat([pd.DataFrame(lr.predict(test_x_data), columns=['predictions']),
                        pd.DataFrame(test_odds_data, columns=['odds']),
                        pd.DataFrame(test_y_data, columns=['isWin']).reset_index()], axis=1)

predict_df['invest'] = predict_df['predictions'] * 100
predict_df['return'] = predict_df['odds'] * predict_df['isWin'] * predict_df['invest'] - predict_df['invest']

print(predict_df.sum())
