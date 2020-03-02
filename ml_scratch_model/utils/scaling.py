import numpy as np
class ScratchStandardScaler():
    """
    標準化する。
    """


    def fit(self, X):
        """
        標準化のために平均と標準偏差を計算する。

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
          学習データ
        """
        self.mean = np.mean(X, axis=0) # 平均の計算
        self.sigma = np.std(X, axis=0) # 分散の計算

    def transform(self, X):
        """
        fitで求めた値を使い標準化を行う。

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
          特徴量

        Returns
        ----------
        X_scaled : 次の形のndarray, shape (n_samples, n_features)
          標準化された特緒量
        """
        X_scaled = (X - self.mean) / self.sigma # 標準化
        
        return X_scaled