import scanpy as sc
from svg_analysis import plot_spatial_gene
import pandas as pd
import matplotlib.pyplot as plt
from adjustText import adjust_text
from matplotlib_venn import venn2
# 加载数据
kidney_file = '/home/zhengyanb/HJ-空转/BMK250409-CR265-ZX01-0101/BMK_DATA_20250804180821_1/03.output/WTkidney_vs_AMLkidney.clustree_result.final.h5ad'
adata = sc.read_h5ad(kidney_file)
outdir = r"/home/zhengyanb/st-analysis/baimaike_result/kidney_WTvsAML/svg_results/"
combined = pd.read_csv(os.path.join(outdir, "svg_combined_by_condition.csv"))
wt_enriched = pd.read_csv(os.path.join(outdir, "svg_0_enriched.csv"))
aml_enriched = pd.read_csv(os.path.join(outdir, "svg_1_enriched.csv"))
diff = pd.read_csv(os.path.join(outdir, "svg_diff_0_vs_1.csv"))
# 绘制WT 和 AML 各自 Top SVG barplot
def plot_top_svg_by_condition(
    combined,
    condition_value,
    condition_name,
    top_n=25,
    outdir=None
):
    df = combined[
        (combined["condition"].astype(str) == str(condition_value)) &
        (combined["q_meta"] < 0.05) &
        (combined["moran_I_weighted_mean"] > 0)
    ].copy()

    df = df.sort_values("z_meta", ascending=False).head(top_n)
    df = df.sort_values("z_meta", ascending=True)

    plt.figure(figsize=(6, max(4, top_n * 0.28)))
    plt.barh(df["gene"], df["z_meta"], color="#4C72B0")
    plt.xlabel("Meta Moran's I Z-score")
    plt.ylabel("")
    plt.title(f"Top {top_n} SVGs in {condition_name}")
    plt.tight_layout()

    if outdir is not None:
        plt.savefig(
            os.path.join(outdir, f"Top_{top_n}_SVG_{condition_name}.png"),
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()


plot_top_svg_by_condition(
    combined,
    condition_value=0,
    condition_name="WT",
    top_n=25,
    outdir=outdir
)

plot_top_svg_by_condition(
    combined,
    condition_value=1,
    condition_name="AML",
    top_n=25,
    outdir=outdir
)

# 绘制图2：WT-enriched SVG barplot
def plot_enriched_svg_bar(
    df,
    enriched_condition_name,
    q_col,
    direction="WT",
    top_n=25,
    outdir=None
):
    data = df.copy()

    # 保险筛选
    if q_col in data.columns:
        data = data[data[q_col] < 0.05]

    if direction == "WT":
        # diff = WT - AML
        data = data[
            (data["delta_moran_I"] > 0) &
            (data["delta_z"] > 0)
        ].copy()
        data = data.sort_values("delta_moran_I", ascending=False).head(top_n)
        color = "#377EB8"
    elif direction == "AML":
        # diff = WT - AML，所以 AML 更强时 delta 为负
        data = data[
            (data["delta_moran_I"] < 0) &
            (data["delta_z"] < 0)
        ].copy()
        data = data.sort_values("delta_moran_I", ascending=True).head(top_n)
        color = "#E41A1C"
    else:
        raise ValueError("direction must be 'WT' or 'AML'")

    data = data.sort_values("delta_moran_I", ascending=True)

    plt.figure(figsize=(6, max(4, top_n * 0.28)))
    plt.barh(data["gene"], data["delta_moran_I"], color=color)
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Delta Moran's I, WT - AML")
    plt.ylabel("")
    plt.title(f"Top {top_n} {enriched_condition_name}-enriched SVGs")
    plt.tight_layout()

    if outdir is not None:
        plt.savefig(
            os.path.join(outdir, f"Top_{top_n}_{enriched_condition_name}_enriched_SVG.png"),
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()


plot_enriched_svg_bar(
    wt_enriched,
    enriched_condition_name="WT",
    q_col="q_0_enriched",
    direction="WT",
    top_n=25,
    outdir=outdir
)

# 绘制图3：AML-enriched SVG barplot
def plot_aml_enriched_svg_correct(
    aml_enriched,
    top_n=25,
    outdir=None
):
    df = aml_enriched.copy()

    # 1. 筛选 AML-enriched SVG
    if "q_1_enriched" in df.columns:
        df = df[df["q_1_enriched"] < 0.05]

    df = df[
        (df["delta_moran_I"] < 0) &
        (df["delta_z"] < 0)
    ].copy()

    # 2. 把 WT - AML 转为 AML - WT
    df["AML_minus_WT_moran_I"] = -df["delta_moran_I"]
    df["AML_minus_WT_z"] = -df["delta_z"]

    # 3. 先按差异强度从大到小选 top
    df_top = df.sort_values(
        "AML_minus_WT_moran_I",
        ascending=False
    ).head(top_n)

    # 4. 为了让 barh 最大的显示在最上方，绘图前按从小到大排序
    df_top = df_top.sort_values(
        "AML_minus_WT_moran_I",
        ascending=True
    )

    # 5. 绘图
    plt.figure(figsize=(6, max(4, top_n * 0.28)))

    plt.barh(
        df_top["gene"],
        df_top["AML_minus_WT_moran_I"],
        color="#E41A1C"
    )

    plt.axvline(0, color="black", linestyle="--", linewidth=1)

    plt.xlabel("Delta Moran's I, AML - WT")
    plt.ylabel("")
    plt.title(f"Top {top_n} AML-enriched SVGs")

    plt.tight_layout()

    if outdir is not None:
        plt.savefig(
            os.path.join(outdir, f"Top_{top_n}_AML_enriched_SVG_correct.png"),
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()

    return df_top
aml_top_plot = plot_aml_enriched_svg_correct(
    aml_enriched=aml_enriched,
    top_n=25,
    outdir=outdir
)

# 绘制图4：WT vs AML differential SVG scatter plot
df = diff.copy()

# WT enriched
wt_sig = df[
    (df["q_0_enriched"] < 0.05) &
    (df["delta_moran_I"] > 0) &
    (df["delta_z"] > 0)
].copy()

# AML enriched
aml_sig = df[
    (df["q_1_enriched"] < 0.05) &
    (df["delta_moran_I"] < 0) &
    (df["delta_z"] < 0)
].copy()

# 选择要标注的 top 基因
top_wt = wt_sig.sort_values("delta_moran_I", ascending=False).head(12)
top_aml = aml_sig.sort_values("delta_moran_I", ascending=True).head(12)

label_df = pd.concat([top_wt, top_aml]).drop_duplicates("gene")

plt.figure(figsize=(7, 6))

# 背景点
plt.scatter(
    df["delta_moran_I"],
    df["delta_z"],
    s=10,
    c="lightgray",
    alpha=0.55,
    edgecolors="none"
)

# WT enriched 点
plt.scatter(
    wt_sig["delta_moran_I"],
    wt_sig["delta_z"],
    s=18,
    c="#377EB8",
    alpha=0.85,
    label="WT-enriched",
    edgecolors="none"
)
# AML enriched 点
plt.scatter(
    aml_sig["delta_moran_I"],
    aml_sig["delta_z"],
    s=18,
    c="#E41A1C",
    alpha=0.85,
    label="AML-enriched",
    edgecolors="none"
)
plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.axvline(0, color="black", linestyle="--", linewidth=1)
texts = []
for _, row in label_df.iterrows():
    color = "#377EB8" if row["delta_moran_I"] > 0 else "#E41A1C"
    texts.append(
        plt.text(
            row["delta_moran_I"],
            row["delta_z"],
            row["gene"],
            fontsize=9,
            color=color,
            fontweight="bold"
        )
    )
adjust_text(
    texts,
    arrowprops=dict(
        arrowstyle="-",
        color="gray",
        lw=0.6,
        alpha=0.7
    ),
    expand_points=(1.5, 1.7),
    expand_text=(1.3, 1.5),
    force_points=0.3,
    force_text=0.7,
    lim=1000
)
plt.xlabel("Delta Moran's I, WT - AML")
plt.ylabel("Delta Z, WT - AML")
plt.title("Differential spatially variable genes: WT vs AML")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(
    os.path.join(outdir, "WT_vs_AML_differential_SVG_scatter_labeled.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.show()

# 绘制图5：WT/AML SVG overlap Venn 图
from matplotlib_venn import venn2

wt_set = set(
    combined[
        (combined["condition"].astype(str) == "0") &
        (combined["q_meta"] < 0.05) &
        (combined["moran_I_weighted_mean"] > 0)
    ]["gene"]
)

aml_set = set(
    combined[
        (combined["condition"].astype(str) == "1") &
        (combined["q_meta"] < 0.05) &
        (combined["moran_I_weighted_mean"] > 0)
    ]["gene"]
)

plt.figure(figsize=(5, 5))
venn2(
    [wt_set, aml_set],
    set_labels=("WT SVGs", "AML SVGs"),
    set_colors=("#377EB8", "#E41A1C"),
    alpha=0.6
)
plt.title("Overlap of SVGs between WT and AML")
plt.tight_layout()

plt.savefig(
    os.path.join(outdir, "WT_AML_SVG_overlap_venn.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.show()

# 绘制图6：关键基因的空间图
# 保存当前表达层
adata.layers["lognorm"] = adata.X.copy()

# 输出目录
sp_ex_dir = os.path.join(outdir, "gene-spatial-expression")
os.makedirs(sp_ex_dir, exist_ok=True)

# genes = [
#     "Ucp1", "Cidea", "Cox8b", "Thrsp", "Plin1", "Fabp4", "Adipoq", "Fasn", "Acaca",
#     "Lipe", "Kap", "Cidec"
# ] # WT 脂肪/产热/脂质代谢模块
# genes = [
#     "Fga", "Fgg", "Serpina1b", "Gc", "Hpd"
# ] # AML 急性期/凝血/血管损伤模块

genes = [
    "Chil3", "Prtn3", "Ifitm1", "Nccrp1", "Iglc1", "Iglc3"
] # AML 免疫/髓系细胞模块

# 两个子集
sub_0 = adata[adata.obs["sample"] == 0].copy()
sub_1 = adata[adata.obs["sample"] == 1].copy()
# 给每个子集配一个名字，保证标题和文件名对应
samples = {
    "WT_kidney": sub_0,
    "AML_kidney": sub_1,
}
for gene in genes:
    for sample_name, sub in samples.items():

        if gene not in sub.var_names:
            print(f"{gene} not found in {sample_name}, skip")
            continue

        title = f"{gene} spatial expression in {sample_name}"
        save_path = os.path.join(
            sp_ex_dir,
            f"{gene}_spatial_expression_in_{sample_name}.png"
        )

        plot_spatial_gene(
            sub,
            gene=gene,
            spatial_key="spatial",
            layer="lognorm",
            title=title,
            save=save_path
        )
