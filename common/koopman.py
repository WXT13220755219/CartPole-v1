import numpy as np
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures

class SindyKoopman:
    def __init__(self, config):
        self.dt = config.dt
        self.poly_order = config.poly_order
        self.threshold = config.threshold  
        self.lasso_alpha = getattr(config, 'lasso_alpha', 1e-5)
        
        # [新增] 是否使用三角函数特征
        self.use_trig = getattr(config, 'use_trig', False)
        
        self.poly = PolynomialFeatures(degree=self.poly_order, include_bias=False)
        
        self.A = None
        self.B = None
        self.n_lifted = None

    def lift(self, X):
        """
        将原始状态 X 映射到高维空间 Z = Psi(X)
        优化: 如果 use_trig=True, 增加 sin(x), cos(x) 特征
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        # 1. 多项式特征
        Z_poly = self.poly.fit_transform(X)
        
        # 2. 三角函数特征 (Pendulum 优化)
        if self.use_trig:
            # 对所有状态计算 sin 和 cos
            Z_sin = np.sin(X)
            Z_cos = np.cos(X)
            # 拼接: [Poly, Sin, Cos]
            return np.hstack([Z_poly, Z_sin, Z_cos])
        else:
            return Z_poly

    def get_feature_names(self):
        """ 获取特征名称 """
        input_features = [f"x{i+1}" for i in range(self.poly.n_features_in_)]
        
        # 多项式名称
        try:
            names = self.poly.get_feature_names_out(input_features).tolist()
        except AttributeError:
            names = self.poly.get_feature_names(input_features)
            
        # 三角函数名称
        if self.use_trig:
            for i in range(len(input_features)):
                names.append(f"sin(x{i+1})")
            for i in range(len(input_features)):
                names.append(f"cos(x{i+1})")
                
        return names

    def fit(self, X, U, X_next):
        """ SINDy 辨识 (与之前逻辑相同，只是 Z 的维度变了) """
        Z = self.lift(X)
        Z_next = self.lift(X_next)
        
        self.n_lifted = Z.shape[1]
        Omega = np.hstack([Z, U])
        
        # Step 1: Lasso 筛选
        lasso = Lasso(alpha=self.lasso_alpha, fit_intercept=False, max_iter=20000)
        lasso.fit(Omega, Z_next)
        K_structure = lasso.coef_ 
        
        # Step 2: Ridge 去偏
        K_final = np.zeros_like(K_structure)
        for i in range(K_structure.shape[0]):
            mask = np.abs(K_structure[i, :]) > 1e-9
            if np.sum(mask) > 0:
                Omega_subset = Omega[:, mask]
                target = Z_next[:, i]
                ols = Ridge(alpha=1e-8, fit_intercept=False)
                ols.fit(Omega_subset, target)
                K_final[i, mask] = ols.coef_
        
        # Step 3: 截断
        K_final[np.abs(K_final) < self.threshold] = 0.0
        
        self.A = K_final[:, :self.n_lifted]
        self.B = K_final[:, self.n_lifted:]
        
        print(f"Koopman Model Identified.")
        print(f"Features: {self.get_feature_names()}")
        print(f"Lifted Dim: {self.n_lifted}, Matrix A: {self.A.shape}")

    def predict(self, z, u):
        return self.A @ z + self.B @ u