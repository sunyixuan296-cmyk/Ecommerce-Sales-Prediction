import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 读取数据（自动找同目录文件）
df = pd.read_csv(os.path.join(os.path.dirname(__file__), "cleaned_data.csv"))

# 日期处理
df['date'] = pd.to_datetime(df['date'])
df['weekday'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month

# 折扣处理
df['discount'] = df['discount'].map({"Yes": 1, "No": 0})

# 类别特征独热编码
df = pd.get_dummies(df, columns=["Store_Type", "Location_Type", "Region_Code"])

# 特征与标签
X = df.drop(columns=["ID", "date", "sales", "order"])
y = df["sales"]

# 划分训练集测试集
split_idx = int(len(df) * 0.7)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# 随机森林模型
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估指标
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n===== 电商销量预测结果 =====")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.4f}")
print("============================")