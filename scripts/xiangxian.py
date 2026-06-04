"""
商店销售数据箱线图分析
功能：对商店销售数据中的关键数值指标绘制箱线图，分析数据分布和异常值
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


def load_and_clean_data(file_path):
    """
    加载并清洗数据
    参数: file_path - CSV文件路径
    返回: 清洗后的DataFrame
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    # 读取数据（处理编码问题）
    try:
        df = pd.read_csv(file_path, encoding='gbk')
        print("成功使用gbk编码读取文件")
    except:
        try:
            df = pd.read_csv(file_path, encoding='gb2312')
            print("成功使用gb2312编码读取文件")
        except:
            df = pd.read_csv(file_path, encoding='latin-1')
            print("成功使用latin-1编码读取文件")

    # 数据清洗
    df_clean = df.iloc[1:].copy()  # 删除第一行中文说明
    if 'Unnamed: 10' in df_clean.columns:
        df_clean = df_clean.drop('Unnamed: 10', axis=1)  # 删除无用列
    df_clean = df_clean.dropna(subset=['ID'])  # 删除ID为空的行

    # 数据类型转换
    numeric_columns = ['Store_id', 'Holiday', 'order', 'sales', 'price']
    for col in numeric_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    # 转换日期列
    if 'Date' in df_clean.columns:
        df_clean['Date'] = pd.to_datetime(df_clean['Date'], errors='coerce')

    # 删除关键数值列的缺失值
    df_clean = df_clean.dropna(subset=['sales', 'order', 'price'])

    print(f"数据清洗完成，有效数据行数：{len(df_clean)}")
    return df_clean


def create_single_boxplot(df, column, title, save_path=None):
    """
    创建单个指标的箱线图
    参数:
        df - 数据DataFrame
        column - 要绘制的列名
        title - 图表标题
        save_path - 保存路径（None则不保存）
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # 创建箱线图
    box_plot = ax.boxplot(df[column],
                          patch_artist=True,
                          notch=True,  # 带缺口的箱线图，显示置信区间
                          vert=True,  # 垂直方向
                          widths=0.6,  # 箱体宽度
                          showmeans=True,  # 显示均值点
                          meanprops=dict(marker='o', markerfacecolor='red', markersize=8, markeredgecolor='darkred'),
                          medianprops=dict(color='darkblue', linewidth=2),
                          boxprops=dict(facecolor='lightblue', alpha=0.7, edgecolor='darkblue', linewidth=1.5),
                          whiskerprops=dict(color='darkblue', linewidth=1.5, linestyle='-'),
                          capprops=dict(color='darkblue', linewidth=1.5),
                          flierprops=dict(marker='x', markerfacecolor='orange', markersize=8, markeredgecolor='red'))

    # 设置图表样式
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel(column, fontsize=12, fontweight='bold')
    ax.set_xticklabels([column], fontsize=12)

    # 添加网格
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    # 添加统计信息文本
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    median = df[column].median()
    mean = df[column].mean()
    std = df[column].std()

    stats_text = f'统计信息：\n'                  f'均值: {mean:.2f}\n'                  f'中位数: {median:.2f}\n'                  f'Q1: {q1:.2f}\n'                  f'Q3: {q3:.2f}\n'                  f'标准差: {std:.2f}'

    ax.text(1.15, 0.95, stats_text, transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
            fontsize=10, verticalalignment='top')

    # 调整布局
    plt.tight_layout()

    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"箱线图已保存至: {save_path}")

    plt.close()


def create_grouped_boxplot(df, value_col, group_col, title, save_path=None):
    """
    创建分组箱线图（按分类变量分组）
    参数:
        df - 数据DataFrame
        value_col - 数值列名
        group_col - 分组列名
        title - 图表标题
        save_path - 保存路径（None则不保存）
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # 获取分组类别
    groups = df[group_col].unique()
    groups = sorted(groups)  # 排序
    data_by_group = [df[df[group_col] == group][value_col] for group in groups]

    # 设置颜色
    colors = ['lightblue', 'lightgreen', 'lightpink', 'lightyellow', 'lightcoral']
    box_colors = colors[:len(groups)]

    # 创建分组箱线图
    box_plot = ax.boxplot(data_by_group,
                          patch_artist=True,
                          notch=True,
                          vert=True,
                          widths=0.6,
                          labels=groups,
                          showmeans=True,
                          meanprops=dict(marker='o', markerfacecolor='red', markersize=7, markeredgecolor='darkred'),
                          medianprops=dict(color='darkblue', linewidth=2),
                          boxprops=dict(edgecolor='darkblue', linewidth=1.5),
                          whiskerprops=dict(color='darkblue', linewidth=1.5),
                          capprops=dict(color='darkblue', linewidth=1.5),
                          flierprops=dict(marker='x', markerfacecolor='orange', markersize=7, markeredgecolor='red'))

    # 为每个箱体设置颜色
    for patch, color in zip(box_plot['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # 设置图表样式
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel(group_col, fontsize=12, fontweight='bold')
    ax.set_ylabel(value_col, fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=11)

    # 添加网格
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    # 添加图例
    legend_elements = [Patch(facecolor=color, edgecolor='darkblue', alpha=0.7, label=f'{group_col}: {group}')
                       for group, color in zip(groups, box_colors)]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    # 调整布局
    plt.tight_layout()

    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"分组箱线图已保存至: {save_path}")

    plt.close()


def create_multi_boxplot(df, columns, title, save_path=None):
    """
    创建多个指标的箱线图（并排显示）
    参数:
        df - 数据DataFrame
        columns - 要绘制的列名列表
        title - 图表标题
        save_path - 保存路径（None则不保存）
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    # 准备数据
    data = [df[col] for col in columns]

    # 设置颜色
    colors = ['lightblue', 'lightgreen', 'lightpink']
    box_colors = colors[:len(columns)]

    # 创建多指标箱线图
    box_plot = ax.boxplot(data,
                          patch_artist=True,
                          notch=True,
                          vert=True,
                          widths=0.6,
                          labels=columns,
                          showmeans=True,
                          meanprops=dict(marker='o', markerfacecolor='red', markersize=8, markeredgecolor='darkred'),
                          medianprops=dict(color='darkblue', linewidth=2),
                          boxprops=dict(edgecolor='darkblue', linewidth=1.5),
                          whiskerprops=dict(color='darkblue', linewidth=1.5),
                          capprops=dict(color='darkblue', linewidth=1.5),
                          flierprops=dict(marker='x', markerfacecolor='orange', markersize=8, markeredgecolor='red'))

    # 为每个箱体设置颜色
    for patch, color in zip(box_plot['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # 设置图表样式
    ax.set_title(title, fontsize=18, fontweight='bold', pad=25)
    ax.set_ylabel('数值', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', labelsize=12, rotation=0)
    ax.tick_params(axis='y', labelsize=12)

    # 添加网格
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    # 添加图例
    legend_elements = [Patch(facecolor='red', label='均值'),
                       Patch(facecolor='darkblue', label='中位数'),
                       Patch(facecolor='orange', label='异常值')]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11)

    # 调整布局
    plt.tight_layout()

    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"多指标箱线图已保存至: {save_path}")

    plt.close()


def main():
    """主函数：执行完整的箱线图分析流程"""
    # 1. 加载和清洗数据
    file_path = 'data/cleaned_data.csv'  # 请根据实际文件路径修改
    try:
        df = load_and_clean_data(file_path)
    except Exception as e:
        print(f"数据加载失败: {str(e)}")
        return

    # 2. 定义关键数值列
    numeric_cols = ['order', 'sales', 'price']

    # 3. 创建单个指标箱线图
    for col in numeric_cols:
        title = f'商店{col}分布箱线图'
        save_path = f'{col}_boxplot.png'
        create_single_boxplot(df, col, title, save_path)

    # 4. 创建分组箱线图（按商店类型分组）
    group_cols = ['Store_Type', 'Location_Type', 'Region_Code']
    for group_col in group_cols:
        for value_col in numeric_cols:
            title = f'按{group_col}分组的{value_col}分布箱线图'
            save_path = f'{value_col}_by_{group_col}_boxplot.png'
            create_grouped_boxplot(df, value_col, group_col, title, save_path)

    # 5. 创建多指标对比箱线图
    title = '商店订单量、销售额、单价对比箱线图'
    save_path = 'multi_indicators_boxplot.png'
    create_multi_boxplot(df, numeric_cols, title, save_path)

    print("\n所有箱线图绘制完成！")
    print("生成的文件包括：")
    print("- 单个指标箱线图：order_boxplot.png, sales_boxplot.png, price_boxplot.png")
    print("- 分组箱线图：按商店类型、地区类型、区域代码分组的箱线图")
    print("- 多指标对比图：multi_indicators_boxplot.png")


if __name__ == "__main__":
    main()
