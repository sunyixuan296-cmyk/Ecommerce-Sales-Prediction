import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# ---------------------- 1. 读取清洗好的数据 ----------------------
df = pd.read_csv("data/cleaned_data.csv")
df['date'] = pd.to_datetime(df['date'])  # 转日期

# ---------------------- 2. 基础统计分析（自动出结论） ----------------------
print("===== 基础数据信息 =====")
print(df.shape)  # 多少行多少列
print(df.describe())  # 销量、价格的平均值、最大最小等

# ---------------------- 3. 销量趋势图（时间序列） ----------------------
plt.figure(figsize=(12,5))
df.groupby('date')['sales'].sum().plot()
plt.title('每日总销量趋势')
plt.ylabel('销量')
plt.show()

# ---------------------- 4. 销量分布直方图（看数据规律） ----------------------
plt.figure(figsize=(8,4))
sns.histplot(df['sales'], bins=30, kde=True)
plt.title('销量分布')
plt.show()

# ---------------------- 5. 价格 vs 销量 散点图（看相关性） ----------------------
# plt.figure(figsize=(8,4))
# sns.scatterplot(x='price', y='sales', data=df)
# plt.title('价格与销量关系')
# plt.show()
plt.figure(figsize=(12, 7))
sns.kdeplot(
    x='price',
    y='sales',
    data=df,
    fill=True,  # 填充颜色，越红表示点越密集
    cmap="Reds",
    thresh=0.05
)
plt.title("价格与销量分布密度", fontsize=16)
plt.show()
# ---------------------- 6. 促销对销量的影响 ----------------------
if 'promo' in df.columns:
    plt.figure(figsize=(6,4))
    sns.boxplot(x='promo', y='sales', data=df)
    plt.title('促销是否影响销量')
    plt.show()

# ---------------------- 7. 节假日销量对比 ----------------------
if 'holiday' in df.columns:
    plt.figure(figsize=(6,4))
    sns.barplot(x='holiday', y='sales', data=df)
    plt.title('节假日 vs 非节假日 销量对比')
    plt.show()

# ---------------------- 8. 相关性热力图（机器学习必用） ----------------------
corr = df[['sales', 'price']].corr()
plt.figure(figsize=(5,4))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('特征相关性')
plt.show()

print("✅ 数据分析 + 可视化全部完成！")
