import numpy as np
import matplotlib.pyplot as plt
from metrics.functions import MSE

class ScratchLogisticRegression():
    """
    ロジスティック回帰
    ＊コンストラクタ（__init__）のパラメータはここに書いておくと分かりやすい

    Parameters
    ----------
    num_iter : int
      イテレーション数
    lr : float
      学習率
    no_bias : bool
      バイアス項を入れない場合はTrue
    verbose : bool
      学習過程を出力する場合はTrue

    Attributes
    ----------
    self.coef_ : 次の形のndarray, shape (n_features,)
      パラメータ
    self.loss : 次の形のndarray, shape (self.iter,)
      学習用データに対する損失の記録
    self.val_loss : 次の形のndarray, shape (self.iter,)
      検証用データに対する損失の記録

    """

    def __init__(self, num_iter, lr, bias, verbose):
        # ハイパーパラメータを属性として記録
        self.num_iter = num_iter
        self.lr = lr
        self.verbose = verbose
        self.n_theta = None
        # 損失を記録する配列を用意
        self.loss = np.zeros(self.num_iter)
        self.val_loss = np.zeros(self.num_iter)
        self._coef = np.zeros(self.num_iter)
        
        
    def _logistic_hypothesis(self, X):
        """
        ロジスティック関数の仮定関数を計算する。

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
          学習データ

        Returns
        -------
          次の形のndarray, shape (n_samples, 1)
          線形の仮定関数による推定結果

        """
        w_x = np.dot(X,self.n_theta)
        sigmoid(w_x)
        return sigmoid(w_x)
    
    def _gradient_descent(self, X, y,lam=0.1):
        """
        最急降下法で重みを更新する。

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
          学習データ
        y : 次の形のndarray, shape (n_samples, 1)
          正解値

        Returns
        -------
          次の形のndarray, shape (1,)

        """
#     self.n_theta = np.random.randn(len(X.columns))
        m = len(X.columns)
        m = float(m)
        for i in range(self.num_iter):
            diff = self._logistic_hypothesis(X).ravel() - y
            self.n_theta = self.n_theta - 0.1 * np.dot((diff - y),xsample)/m + lam * self.n_theta / m
        
        return self.n_theta
        
    def _compute_cost(self, X, y):
        """
        クロスエントロピー誤差を計算する。クロスエントロピー誤差は共通の関数を作っておき呼び出す

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
          学習データ
        y : 次の形のndarray, shape (n_samples, 1)
          正解値

        Returns
        -------
          次の形のndarray, shape (1,)
          クロスエントロピー
        """
        y_pred = self._logistic_hypothesis(X).ravel()
        print(y_pred)
        return cross_entropy(y_pred, y)
    
        

    def fit(self, X, y, X_val=None, y_val=None):
        """
        ロジスティック回帰を学習する。
        検証用データが入力された場合はそれに対する損失と精度もイテレーションごとに計算する。

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            学習用データの特徴量
        y : 次の形のndarray, shape (n_samples, )
            学習用データの正解値
        X_val : 次の形のndarray, shape (n_samples, n_features)
            検証用データの特徴量
        y_val : 次の形のndarray, shape (n_samples, )
            検証用データの正解値
        """
        self.n_theta = np.random.randn(len(X.columns))
        for i in range(self.num_iter):
            self._gradient_descent(X, y)
            loss = self._compute_cost(X, y)
            self.loss[i] = loss
            if (X_val is not None) and (y_val is not None):
                val_loss = self._compute_cost(X_val, y_val)
                self.val_loss[i] = val_loss
            else:
                self.val_loss[i] = None
                
            plt.subplot(1,2,1)
            plt.plot(i,self.n_theta[0], marker='o')
            plt.subplot(1,2,2)
            plt.plot(i, self.n_theta[1], marker='o')
            plt.subplots_adjust(wspace=0.5)
            
            if self.verbose:
                #verboseをTrueにした際は学習過程を出力
                print('[{}]loss:   {} val_loss: {}'.format(i,self.loss[i], self.val_loss[i]))
                print('[{}]coef:   {}'.format(i,self.n_theta))
                print('-'*50)
                
        plt.show()


    def predict(self, X):
        """
        線形回帰を使い推定する。

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            サンプル

        Returns
        -------
            次の形のndarray, shape (n_samples, 1)
            線形回帰による推定結果
        """

        predict_proba = self._logistic_hypothesis(X)
        predict_class = [1 if i>=0.5 else 0 for i in predict_proba]
        
        return predict_class