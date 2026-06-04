import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

csv_path = r"data\cleaned_data.csv"
df = pd.read_csv(csv_path)

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

# import pandas as pd
# import numpy as np
# import os
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# # ===================== 固定路径（必能运行）=====================
# csv_path = r"C:\Users\dengshifeng\Desktop\Ecommerce-Sales-Prediction\data\cleaned_data.csv"
# df = pd.read_csv(csv_path)

# # 日期处理
# df['date'] = pd.to_datetime(df['date'])
# df['weekday'] = df['date'].dt.dayofweek
# df['month'] = df['date'].dt.month

# # 折扣处理
# df['discount'] = df['discount'].map({"Yes": 1, "No": 0})

# # 类别特征独热编码
# df = pd.get_dummies(df, columns=["Store_Type", "Location_Type", "Region_Code"])

# # 特征与标签
# X = df.drop(columns=["ID", "date", "sales", "order"])
# y = df["sales"]

# # 划分训练集测试集
# split_idx = int(len(df) * 0.7)
# X_train, X_test = X[:split_idx], X[split_idx:]
# y_train, y_test = y[:split_idx], y[split_idx:]

# # ===================== 1. 多元线性回归 =====================
# lr = LinearRegression()
# lr.fit(X_train, y_train)
# lr_pred = lr.predict(X_test)

# lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
# lr_mae = mean_absolute_error(y_test, lr_pred)
# lr_r2 = r2_score(y_test, lr_pred)

# # ===================== 2. 随机森林回归 =====================
# rf = RandomForestRegressor(
#     n_estimators=100,
#     random_state=42,
#     n_jobs=-1
# )
# rf.fit(X_train, y_train)
# rf_pred = rf.predict(X_test)

# rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
# rf_mae = mean_absolute_error(y_test, rf_pred)
# rf_r2 = r2_score(y_test, rf_pred)

# # ===================== 输出对比结果 =====================
# print("=== 电商销量预测——双模型对比结果 ===\n")
# print("【多元线性回归】")
# print(f"RMSE: {lr_rmse:.2f}")
# print(f"MAE:  {lr_mae:.2f}")
# print(f"R²:   {lr_r2:.4f}\n")

# print("【随机森林回归】")
# print(f"RMSE: {rf_rmse:.2f}")
# print(f"MAE:  {rf_mae:.2f}")
# print(f"R²:   {rf_r2:.4f}\n")
# print("===================================")
