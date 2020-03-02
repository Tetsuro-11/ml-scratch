import argparse
import pandas as pd

# モジュールを読み込む
from model.linear_regression import ScratchLinearRegression
from utils.split import train_test_split
from utils.scaling import ScratchStandardScaler

# コマンドライン引数の設定
parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description='logistic_regression')

parser.add_argument('--iter', default=10, type=int,
                    help='number of iterations')
parser.add_argument('--lr', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--no_bias', action='store_true',
                    help='without bias')
parser.add_argument('--verbose', action='store_true',
                    help='output of learning process')
parser.add_argument('--dataset', default='train.csv', type=str,
                    help='path to csvfile')

def main():
    # コマンドライン引数の読み込み
    args = parser.parse_args()
    # データセットの読み込み
    train = pd.read_csv(args.dataset)
    

    # 前処理
#     x_train = train.drop('SalePrice', axis=1)
#     x_train = train.loc[:, ["GrLivArea", "YearBuilt"]]
#     scaler = ScratchStandardScaler()
#     scaler.fit(x_train)
#     x_train = scaler.transform(x_train)
#     y_train = train['SalePrice']
#     x_train, y_train, X_val, y_val = train_test_split(x_train, y_train, train_size=0.8)
    
    x_train = train.drop('SalePrice', axis=1)
    y_train = train['SalePrice']
    scaler = ScratchStandardScaler()
    scaler.fit(x_train[["GrLivArea", "YearBuilt"]])
    x_train =scaler.transform(x_train[["GrLivArea", "YearBuilt"]])
    X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train, train_size = 0.8)

    model = ScratchLinearRegression(args.iter, args.lr, args.no_bias, args.verbose)
    model.fit(X_train, Y_train, X_val, Y_val)

    train_loss = model.loss
    val_loss = model.val_loss

    y_pred = model.predict(X_val)

if __name__ == '__main__':
    main()