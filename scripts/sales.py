import pandas as pd
import numpy as np

# ====================== 1. 读取数据（你只需要改这里的文件名） ======================
df = pd.read_csv("TRAIN.csv", encoding='gbk')  # 替换成你的csv文件名
print("原始数据形状：", df.shape)

# ====================== 2. 统一数据格式 ======================
# 日期列统一格式（必须转，否则无法做时间预测）
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')  # 错误转NaT

# 销量、价格必须是数字类型
for col in ['sales', 'quantity', 'price', 'item_price', 'promo']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# 商品ID统一为字符串（避免ID变成数字）
for col in ['item_id', 'product_id', 'sku_id', 'store_id']:
    if col in df.columns:
        df[col] = df[col].astype(str)

print("✅ 格式统一完成")

# ====================== 3. 删除重复数据 ======================
before = df.shape[0]
df = df.drop_duplicates()
after = df.shape[0]
print(f"✅ 删除重复数据：{before - after} 条")

# ====================== 4. 处理缺失值 ======================
print("\n缺失值统计（前10列）：")
print(df.isnull().sum().head(10))

# 数值型缺失 → 用中位数填充（销量预测最稳）
num_cols = df.select_dtypes(include=['number']).columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# 类别/ID缺失 → 直接删除行（因为商品ID不能缺）
cat_cols = ['item_id', 'product_id', 'date']
for col in cat_cols:
    if col in df.columns:
        df = df.dropna(subset=[col])

print("✅ 缺失值处理完成")

# ====================== 5. 异常值处理（销量预测专用） ======================
# 方法：3σ原则（超过3倍标准差视为异常）
def remove_outliers(df, col):
    if col not in df.columns:
        return df
    mean = df[col].mean()
    std = df[col].std()
    df = df[(df[col] > mean - 3*std) & (df[col] < mean + 3*std)]
    return df

# 只对销量、价格做异常值过滤
df = remove_outliers(df, 'sales')
df = remove_outliers(df, 'price')

print("✅ 异常值处理完成")
print("清洗后数据形状：", df.shape)

# ====================== 6. 保存清洗后数据 ======================
df.to_csv("cleaned_data.csv", index=False)
print("\n🎉 数据清洗全部完成！文件已保存为 cleaned_data.csv")