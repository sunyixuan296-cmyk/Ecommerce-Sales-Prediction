import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------- 全局字体/样式设置 --------------------------
plt.rcParams['font.sans-serif'] = ['SimHei']       # 中文黑体
plt.rcParams['axes.unicode_minus'] = False         # 正常显示负号
plt.rcParams['xtick.labelsize'] = 14                # x刻度数字大小
plt.rcParams['ytick.labelsize'] = 14                # y刻度数字大小
plt.rcParams['font.weight'] = 'bold'                # 全局数字粗体

# 读取数据
df = pd.read_csv("cleaned_data.csv")

# ====================== 图1：销量分布（正方形、去上/左边框） ======================
fig, ax = plt.subplots(figsize=(8, 8))  # 正方形画布
ax.hist(df["sales"], bins=20, color="skyblue", edgecolor="black")
ax.set_title("商品销量分布", fontsize=16, fontweight='bold')
ax.set_xlabel("销量", fontsize=14, fontweight='bold')
ax.set_ylabel("频次", fontsize=14, fontweight='bold')

# 去掉上、左边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_box_aspect(1)  # 强制正方形坐标轴
plt.tight_layout()
plt.savefig("fig1_sales_dist.png", dpi=300)
plt.close()

# ====================== 图2：价格与销量关系（六边形热力图） ======================
fig, ax = plt.subplots(figsize=(8, 8))
hb = ax.hexbin(df['price'], df['sales'], gridsize=30, cmap='Blues')
cbar = plt.colorbar(hb, ax=ax)
cbar.set_label('数据点数量', fontsize=14, fontweight='bold')
cbar.ax.tick_params(labelsize=14)

ax.set_title('价格与销量关系（六边形热力图）', fontsize=16, fontweight='bold')
ax.set_xlabel('价格', fontsize=14, fontweight='bold')
ax.set_ylabel('销量', fontsize=14, fontweight='bold')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_box_aspect(1)

plt.tight_layout()
plt.savefig("fig2_price_vs_sales_hex.png", dpi=300)
plt.close()

# ====================== 图3：相关性热力图 ======================
corr = df[["sales", "price"]].corr()

fig, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax,
            annot_kws={"size": 14, "weight": "bold"},  # 相关系数14号粗体
            cbar_kws={"shrink": .8})

ax.set_title("特征相关性热力图", fontsize=16, fontweight='bold')
ax.set_xlabel("")
ax.set_ylabel("")

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_box_aspect(1)

plt.tight_layout()
plt.savefig("fig3_correlation.png", dpi=300)
plt.close()

print("✅ 全部成功！三张正方形图已生成（数字14号粗体、已去上/左边框）！")