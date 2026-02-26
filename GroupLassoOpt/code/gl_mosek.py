import numpy as np
import mosek
from mosek.fusion import Model, Domain, Expr, ObjectiveSense, Matrix

def gl_mosek(x0: np.ndarray, A: np.ndarray, b: np.ndarray, mu: float, opts:dict = {}):
    
    m, n = A.shape
    l = b.shape[1]
    
    # 创建 MOSEK Fusion 模型
    with Model('GroupLASSO_MOSEK') as M:
        # 关闭日志输出
        M.setLogHandler(None)
        
        # 将 numpy 数组转换为 MOSEK 矩阵
        A_mat = Matrix.dense(A)
        b_mat = Matrix.dense(b)
        
        # 变量 X (n x l)
        X = M.variable("X", [n, l], Domain.unbounded())
        
        # 变量 Y (m x l) 用于残差
        Y = M.variable("Y", [m, l], Domain.unbounded())
        
        # 变量 t0 用于 ||Y||_F^2 的旋转锥表示
        t0 = M.variable(1, Domain.unbounded())
        
        # 变量 ts (n) 用于组范数
        ts = M.variable(n, Domain.greaterThan(0.0))
        
        # 约束: Y = A X - b
        M.constraint(
            Expr.sub(Expr.sub(Expr.mul(A_mat, X), b_mat), Y),
            Domain.equalsTo(0.0)
        )
        
        # 旋转二次锥约束: t0 >= 0.5 * ||Y||_F^2
        # 等价于: (t0, 1, vec(Y)) 在旋转二次锥中
        Y_vec = Y.reshape(m * l)
        M.constraint(
            Expr.vstack(t0, Expr.constTerm(1, 1.0), Y_vec),
            Domain.inRotatedQCone()
        )
        
        # 二阶锥约束: ts_i >= ||X[i,:]||_2
        # 需要将 X 的行转换为向量
        for i in range(n):
            # 获取 X 的第 i 行，然后转换为向量
            X_row = X.slice([i, 0], [i+1, l])
            X_row_vec = Expr.reshape(X_row, l)
            
            # 二阶锥约束: (ts_i, X_row) 在二次锥中
            M.constraint(
                Expr.vstack(ts.index(i), X_row_vec),
                Domain.inQCone()
            )
        
        # 目标函数: t0 + mu * sum(ts)
        obj = Expr.add(t0, Expr.mul(mu, Expr.sum(ts)))
        M.objective('obj', ObjectiveSense.Minimize, obj)
        
        # 求解
        M.solve()
        
        # 获取解
        x_opt = X.level().reshape(n, l)
        opt_value = M.primalObjValue()
        
        return x_opt, -1, [opt_value]

