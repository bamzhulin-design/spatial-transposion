import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.stats import norm
from sklearn.neighbors import NearestNeighbors
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ============================================================
# 1. 基础工具函数
# ============================================================

def make_output_dir(outdir):
    os.makedirs(outdir, exist_ok=True
    return outdir

def infer_key(adata, candidates, key_type="obs"):
    """
    自动寻找 obs 或 obsm 中可能存在的 key。
    """
    if key_type == "obs":
        keys = adata.obs.columns
    elif key_type == "obsm":
        keys = adata.obsm.keys()
    else:
        raise ValueError("key_type must be 'obs' or 'obsm'")

    for k in candidates:
        if k in keys:
            return k
    return None

def get_expression_matrix(adata, layer=None):
    """
    获取表达矩阵。
    默认使用 adata.X，也可以指定 layer。
    """
    if layer is None:
        X = adata.X
    else:
        X = adata.layers[layer]

    if sp.issparse(X):
        X = X.toarray()

    return np.asarray(X, dtype=np.float32)

def row_normalize_sparse_matrix(W):
    """
    对稀疏矩阵按行归一化。
    """
    row_sum = np.asarray(W.sum(axis=1)).ravel()
    row_sum[row_sum == 0] = 1.0
    inv_row_sum = 1.0 / row_sum
    D_inv = sp.diags(inv_row_sum)
    return D_inv @ W

# ============================================================
# 2. 空间邻接图构建
# ============================================================

def build_spatial_knn_graph(coords, k=6, symmetric=True, row_normalize=True):
    """
    基于空间坐标构建 kNN 空间图。

    Parameters
    ----------
    coords:
        n_spots x 2 或 n_spots x d 的空间坐标。
    k:
        每个 spot 的空间邻居数。
    symmetric:
        是否将 kNN 图对称化。
    row_normalize:
        是否行归一化。

    Returns
    -------
    W:
        scipy sparse csr_matrix, n_spots x n_spots
    """
    n = coords.shape[0]
    if n <= k:
        k = max(1, n - 1)

    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(coords)
    distances, indices = nn.kneighbors(coords)

    # 去掉自己，即第一个邻居
    indices = indices[:, 1:]
    distances = distances[:, 1:]

    rows = np.repeat(np.arange(n), k)
    cols = indices.reshape(-1)

    # 这里使用 binary weights。也可以改成距离权重。
    data = np.ones(len(rows), dtype=np.float32)

    W = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()

    if symmetric:
        W = W.maximum(W.T)

    W.setdiag(0)
    W.eliminate_zeros()

    if row_normalize:
        W = row_normalize_sparse_matrix(W)

    return W.tocsr()

# ============================================================
# 3. Moran's I 计算
# ============================================================

def morans_i_vectorized(X, W, chunk_size=1000):
    """
    向量化计算所有基因的 Moran's I。

    Parameters
    ----------
    X:
        n_spots x n_genes 表达矩阵，建议为 log-normalized 表达。
    W:
        n_spots x n_spots 空间权重矩阵，推荐行归一化。
    chunk_size:
        分块计算基因数量，避免内存过高。

    Returns
    -------
    I:
        每个基因的 Moran's I。
    """
    n_cells, n_genes = X.shape
    I_all = np.zeros(n_genes, dtype=np.float64)

    for start in range(0, n_genes, chunk_size):
        end = min(start + chunk_size, n_genes)
        Xc = X[:, start:end].astype(np.float64)

        # 每个基因中心化
        Z = Xc - Xc.mean(axis=0, keepdims=True)

        denom = np.sum(Z ** 2, axis=0)
        denom[denom == 0] = np.nan

        WZ = W @ Z
        numerator = np.sum(Z * WZ, axis=0)

        # 如果 W 行归一化且每行和约为 1，则 S0 约等于 n，Moran's I = numerator / denominator
        I = numerator / denom
        I_all[start:end] = I

    return I_all

def morans_i_permutation_test(
    X,
    W,
    genes,
    n_perm=199,
    seed=123,
    chunk_size=500,
    alternative="greater"
):
    """
    对所有基因计算 Moran's I 并做置换检验。

    Parameters
    ----------
    X:
        n_spots x n_genes log-normalized 表达矩阵。
    W:
        空间邻接矩阵。
    genes:
        gene name list。
    n_perm:
        置换次数。正式分析建议 999 或更高；快速测试可以 99/199。
    seed:
        随机种子。
    chunk_size:
        分块大小。
    alternative:
        "greater" 表示检测正空间自相关 SVG。

    Returns
    -------
    df:
        包含 gene, moran_I, pval, qval, z_score 的 DataFrame。
    """
    rng = np.random.default_rng(seed)
    n_cells, n_genes = X.shape

    obs_I = morans_i_vectorized(X, W, chunk_size=chunk_size)

    perm_sum = np.zeros(n_genes, dtype=np.float64)
    perm_sumsq = np.zeros(n_genes, dtype=np.float64)
    extreme_count = np.zeros(n_genes, dtype=np.int64)

    for b in tqdm(range(n_perm), desc="Permutation test"):
        perm_idx = rng.permutation(n_cells)
        X_perm = X[perm_idx, :]

        perm_I = morans_i_vectorized(X_perm, W, chunk_size=chunk_size)

        perm_sum += perm_I
        perm_sumsq += perm_I ** 2

        if alternative == "greater":
            extreme_count += perm_I >= obs_I
        elif alternative == "less":
            extreme_count += perm_I <= obs_I
        else:
            extreme_count += np.abs(perm_I) >= np.abs(obs_I)

    perm_mean = perm_sum / n_perm
    perm_var = perm_sumsq / n_perm - perm_mean ** 2
    perm_sd = np.sqrt(np.maximum(perm_var, 1e-12))

    z_score = (obs_I - perm_mean) / perm_sd

    pval = (extreme_count + 1) / (n_perm + 1)
    qval = multipletests(pval, method="fdr_bh")[1]

    df = pd.DataFrame({
        "gene": genes,
        "moran_I": obs_I,
        "perm_mean": perm_mean,
        "perm_sd": perm_sd,
        "z_score": z_score,
        "pval": pval,
        "qval": qval,
    })

    df = df.sort_values(["qval", "moran_I"], ascending=[True, False]).reset_index(drop=True)
    return df

# ============================================================
# 4. 单样本 SVG 分析
# ============================================================

def run_svg_for_one_sample(
    adata,
    sample_name,
    condition_name,
    spatial_key="spatial",
    layer="lognorm",
    k=6,
    n_perm=199,
    min_cells_expressed=20,
    seed=123,
    chunk_size=500,
):
    """
    对一个样本计算 SVG。
    """
    coords = adata.obsm[spatial_key]

    X = get_expression_matrix(adata, layer=layer)
    genes = np.asarray(adata.var_names)

    # 过滤低表达基因
    expressed_cells = np.sum(X > 0, axis=0)
    keep = expressed_cells >= min_cells_expressed

    X = X[:, keep]
    genes = genes[keep]
    expressed_cells = expressed_cells[keep]

    if X.shape[1] == 0:
        raise ValueError(f"Sample {sample_name} has no genes after filtering.")

    W = build_spatial_knn_graph(coords, k=k, symmetric=True, row_normalize=True)

    df = morans_i_permutation_test(
        X=X,
        W=W,
        genes=genes,
        n_perm=n_perm,
        seed=seed,
        chunk_size=chunk_size,
        alternative="greater",
    )

    df["sample"] = sample_name
    df["condition"] = condition_name
    df["n_spots"] = adata.n_obs
    df["n_expressed_cells"] = df["gene"].map(
        dict(zip(genes, expressed_cells))
    )

    return df

# ============================================================
# 5. 多样本结果合并
# ============================================================

def combine_svg_by_condition(sample_results):
    """
    将多个样本的 SVG 结果按 condition 合并。

    使用加权 Stouffer Z 方法：
        z_meta = sum(w_i * z_i) / sqrt(sum(w_i^2))
    权重使用 sqrt(n_spots)。

    Returns
    -------
    combined_df
    """
    all_df = pd.concat(sample_results, axis=0, ignore_index=True)

    combined = []

    for condition, sub_cond in all_df.groupby("condition"):
        genes = sorted(sub_cond["gene"].unique())

        for gene in genes:
            sub = sub_cond[sub_cond["gene"] == gene].copy()

            z = sub["z_score"].values.astype(float)
            w = np.sqrt(sub["n_spots"].values.astype(float))

            valid = np.isfinite(z)
            z = z[valid]
            w = w[valid]
            sub_valid = sub.iloc[np.where(valid)[0]]

            if len(z) == 0:
                continue

            z_meta = np.sum(w * z) / np.sqrt(np.sum(w ** 2))
            p_meta = norm.sf(z_meta)

            moran_mean = np.average(
                sub_valid["moran_I"].values,
                weights=np.sqrt(sub_valid["n_spots"].values)
            )

            combined.append({
                "condition": condition,
                "gene": gene,
                "n_samples": sub_valid["sample"].nunique(),
                "samples": ",".join(sub_valid["sample"].astype(str).unique()),
                "moran_I_weighted_mean": moran_mean,
                "z_meta": z_meta,
                "p_meta": p_meta,
            })

    combined_df = pd.DataFrame(combined)

    qvals = []
    for condition, sub in combined_df.groupby("condition"):
        q = multipletests(sub["p_meta"].values, method="fdr_bh")[1]
        tmp = pd.Series(q, index=sub.index)
        qvals.append(tmp)

    combined_df["q_meta"] = pd.concat(qvals).sort_index()

    combined_df = combined_df.sort_values(
        ["condition", "q_meta", "moran_I_weighted_mean"],
        ascending=[True, True, False]
    ).reset_index(drop=True)

    return combined_df, all_df

# ============================================================
# 6. AML vs WT 空间变异差异比较
# ============================================================

def compare_conditions_svg(combined_df, condition_a="AML", condition_b="WT"):
    """
    比较两个 condition 的 SVG 强度。

    这里基于 meta z-score 做探索性比较：
        diff_z = z_A - z_B
        p_A_enriched = P(Z > diff_z / sqrt(2))

    注意：
    如果每组有多个生物学重复，最好进一步对样本级 moran_I 做 replicate-level 统计。
    """
    a = combined_df[combined_df["condition"] == condition_a].copy()
    b = combined_df[combined_df["condition"] == condition_b].copy()

    a = a.set_index("gene")
    b = b.set_index("gene")

    genes = sorted(set(a.index).intersection(set(b.index)))

    rows = []
    for gene in genes:
        za = a.loc[gene, "z_meta"]
        zb = b.loc[gene, "z_meta"]

        Ia = a.loc[gene, "moran_I_weighted_mean"]
        Ib = b.loc[gene, "moran_I_weighted_mean"]

        diff_z = za - zb
        diff_I = Ia - Ib

        # 两个独立标准化 z 的差异，方差近似为 2
        z_diff_std = diff_z / np.sqrt(2)

        p_a_enriched = norm.sf(z_diff_std)
        p_b_enriched = norm.cdf(z_diff_std)

        rows.append({
            "gene": gene,
            f"{condition_a}_z_meta": za,
            f"{condition_b}_z_meta": zb,
            f"{condition_a}_moran_I": Ia,
            f"{condition_b}_moran_I": Ib,
            "delta_z": diff_z,
            "delta_moran_I": diff_I,
            f"p_{condition_a}_enriched": p_a_enriched,
            f"p_{condition_b}_enriched": p_b_enriched,
        })

    df = pd.DataFrame(rows)

    df[f"q_{condition_a}_enriched"] = multipletests(
        df[f"p_{condition_a}_enriched"].values,
        method="fdr_bh"
    )[1]

    df[f"q_{condition_b}_enriched"] = multipletests(
        df[f"p_{condition_b}_enriched"].values,
        method="fdr_bh"
    )[1]

    df = df.sort_values("delta_z", ascending=False).reset_index(drop=True)

    return df

# ============================================================
# 7. 可视化
# ============================================================

def plot_spatial_gene(
    adata,
    gene,
    spatial_key="spatial",
    layer="lognorm",
    color_map="viridis",
    title=None,
    save=None,
    point_size=12,
):
    """
    不使用 squidpy 的空间基因表达图。
    """
    if gene not in adata.var_names:
        raise ValueError(f"{gene} not found in adata.var_names")

    coords = adata.obsm[spatial_key]
    gene_idx = np.where(adata.var_names == gene)[0][0]

    X = get_expression_matrix(adata, layer=layer)
    expr = X[:, gene_idx]

    plt.figure(figsize=(5, 5))
    plt.scatter(
        coords[:, 0],
        coords[:, 1],
        c=expr,
        s=point_size,
        cmap=color_map,
        edgecolor="none"
    )
    plt.gca().invert_yaxis()
    plt.axis("equal")
    plt.axis("off")
    plt.colorbar(label=gene)

    if title is None:
        title = gene
    plt.title(title)

    if save is not None:
        plt.savefig(save, dpi=300, bbox_inches="tight")

    plt.show()

def plot_top_svg_bar(combined_df, condition, top_n=30, save=None):
    """
    绘制某个 condition 的 top SVG barplot。
    """
    sub = combined_df[combined_df["condition"] == condition].copy()
    sub = sub.sort_values("z_meta", ascending=False).head(top_n)

    plt.figure(figsize=(6, max(4, top_n * 0.25)))
    sns.barplot(
        data=sub,
        x="z_meta",
        y="gene",
        color="#4C72B0"
    )
    plt.xlabel("Meta Moran's I Z-score")
    plt.ylabel("")
    plt.title(f"Top {top_n} SVGs in {condition}")
    plt.tight_layout()

    if save is not None:
        plt.savefig(save, dpi=300, bbox_inches="tight")

    plt.show()


# ============================================================
# 8. 主流程
# ============================================================

def run_full_svg_pipeline(
    h5ad_path,
    outdir="./svg_no_squidpy_results",
    condition_col=None,
    sample_col=None,
    spatial_key=None,
    condition_a="AML",
    condition_b="WT",
    k=6,
    n_perm=199,
    n_top_hvg=3000,
    min_cells_expressed=20,
    seed=123,
):
    """
    完整 SVG 分析流程。

    Parameters
    ----------
    h5ad_path:
        输入 h5ad 文件。
    outdir:
        输出目录。
    condition_col:
        AML/WT 信息所在的 obs 列名。
        如果不填，会尝试自动识别。
    sample_col:
        样本或切片 ID 所在的 obs 列名。
        如果不填，会尝试自动识别。
    spatial_key:
        空间坐标 obsm key。
        如果不填，会尝试使用 obsm['spatial']。
    condition_a:
        比较时的第一个 condition，例如 AML。
    condition_b:
        比较时的第二个 condition，例如 WT。
    k:
        空间 kNN 邻居数。Visium 常用 6。
    n_perm:
        Moran's I 置换次数。正式分析建议 999。
    n_top_hvg:
        用于 SVG 检验的高变基因数。
        空间分析不建议对所有 2 万基因做置换，太慢。
    min_cells_expressed:
        基因至少在多少 spots 中表达。
    seed:
        随机种子。
    """
    make_output_dir(outdir)

    print("Loading h5ad...")
    adata = sc.read_h5ad(h5ad_path)
    print(adata)

    # 自动识别 spatial key
    if spatial_key is None:
        spatial_key = infer_key(
            adata,
            candidates=["spatial", "X_spatial", "spatial_coords"],
            key_type="obsm"
        )

    if spatial_key is None:
        raise ValueError(
            "Cannot find spatial coordinates. Please set spatial_key manually, e.g. spatial_key='spatial'."
        )

    print(f"Using spatial coordinates: adata.obsm['{spatial_key}']")

    # 自动识别 condition column
    if condition_col is None:
        condition_col = infer_key(
            adata,
            candidates=[
                "condition", "Condition", "group", "Group",
                "disease", "Disease", "genotype", "Genotype",
                "status", "Status", "type", "Type"
            ],
            key_type="obs"
        )

    if condition_col is None:
        raise ValueError(
            "Cannot infer condition column. Please set condition_col manually, e.g. condition_col='condition'."
        )

    print(f"Using condition column: adata.obs['{condition_col}']")

    # 自动识别 sample column
    if sample_col is None:
        sample_col = infer_key(
            adata,
            candidates=[
                "sample", "Sample", "sample_id", "SampleID",
                "library_id", "slice", "slice_id", "batch", "Batch"
            ],
            key_type="obs"
        )

    if sample_col is None:
        print("No sample column found. Treating all spots as one sample.")
        adata.obs["sample_auto"] = "sample_1"
        sample_col = "sample_auto"

    print(f"Using sample column: adata.obs['{sample_col}']")

    # 基础检查
    print("Condition distribution:")
    print(adata.obs[condition_col].value_counts())

    print("Sample distribution:")
    print(adata.obs[sample_col].value_counts())

    # 确保 condition 是字符串
    adata.obs[condition_col] = adata.obs[condition_col].astype(str)
    adata.obs[sample_col] = adata.obs[sample_col].astype(str)

    # ------------------------------------------------------------
    # 预处理
    # ------------------------------------------------------------
    print("Preprocessing...")

    # 保存原始 counts
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()

    # Normalize + log1p
    adata_pp = adata.copy()
    sc.pp.normalize_total(adata_pp, target_sum=1e4)
    sc.pp.log1p(adata_pp)
    adata_pp.layers["lognorm"] = adata_pp.X.copy()

    # 选择 HVG
    print(f"Selecting top {n_top_hvg} highly variable genes...")
    sc.pp.highly_variable_genes(
        adata_pp,
        n_top_genes=n_top_hvg,
        flavor="seurat",
        layer="lognorm"
    )

    hvg_genes = adata_pp.var_names[adata_pp.var["highly_variable"]].tolist()
    print(f"Number of HVGs: {len(hvg_genes)}")

    adata_hvg = adata_pp[:, hvg_genes].copy()

    # ------------------------------------------------------------
    # 每个样本分别计算 SVG
    # ------------------------------------------------------------
    sample_results = []

    samples = sorted(adata_hvg.obs[sample_col].unique())

    for i, sample in enumerate(samples):
        sub = adata_hvg[adata_hvg.obs[sample_col] == sample].copy()

        if sub.n_obs < 30:
            print(f"Skipping sample {sample}: too few spots.")
            continue

        conds = sub.obs[condition_col].unique()
        if len(conds) != 1:
            print(
                f"Warning: sample {sample} contains multiple conditions: {conds}. "
                f"Will use the most frequent condition."
            )

        condition_name = sub.obs[condition_col].value_counts().index[0]

        print("=" * 80)
        print(f"Running SVG for sample {sample}, condition {condition_name}, spots={sub.n_obs}")
        print("=" * 80)

        df_sample = run_svg_for_one_sample(
            adata=sub,
            sample_name=sample,
            condition_name=condition_name,
            spatial_key=spatial_key,
            layer="lognorm",
            k=k,
            n_perm=n_perm,
            min_cells_expressed=min_cells_expressed,
            seed=seed + i,
            chunk_size=500,
        )

        sample_out = os.path.join(outdir, f"svg_sample_{sample}.csv")
        df_sample.to_csv(sample_out, index=False)
        print(f"Saved: {sample_out}")

        sample_results.append(df_sample)

    if len(sample_results) == 0:
        raise ValueError("No valid sample results generated.")

    # ------------------------------------------------------------
    # 合并 condition-level SVG
    # ------------------------------------------------------------
    print("Combining sample-level SVG results by condition...")
    combined_df, all_sample_df = combine_svg_by_condition(sample_results)

    all_sample_path = os.path.join(outdir, "svg_all_samples.csv")
    combined_path = os.path.join(outdir, "svg_combined_by_condition.csv")

    all_sample_df.to_csv(all_sample_path, index=False)
    combined_df.to_csv(combined_path, index=False)

    print(f"Saved: {all_sample_path}")
    print(f"Saved: {combined_path}")

    # ------------------------------------------------------------
    # AML vs WT 对比
    # ------------------------------------------------------------
    print(f"Comparing {condition_a} vs {condition_b} SVG patterns...")

    diff_df = compare_conditions_svg(
        combined_df,
        condition_a=condition_a,
        condition_b=condition_b
    )

    diff_path = os.path.join(outdir, f"svg_diff_{condition_a}_vs_{condition_b}.csv")
    diff_df.to_csv(diff_path, index=False)
    print(f"Saved: {diff_path}")

    # ------------------------------------------------------------
    # 输出显著 SVG
    # ------------------------------------------------------------
    sig_combined = combined_df[combined_df["q_meta"] < 0.05].copy()
    sig_combined_path = os.path.join(outdir, "svg_significant_by_condition_q005.csv")
    sig_combined.to_csv(sig_combined_path, index=False)

    print(f"Saved significant SVGs: {sig_combined_path}")

    aml_enriched = diff_df[
        diff_df[f"q_{condition_a}_enriched"] < 0.05
    ].sort_values("delta_z", ascending=False)

    wt_enriched = diff_df[
        diff_df[f"q_{condition_b}_enriched"] < 0.05
    ].sort_values("delta_z", ascending=True)

    aml_path = os.path.join(outdir, f"svg_{condition_a}_enriched.csv")
    wt_path = os.path.join(outdir, f"svg_{condition_b}_enriched.csv")

    aml_enriched.to_csv(aml_path, index=False)
    wt_enriched.to_csv(wt_path, index=False)

    print(f"Saved {condition_a}-enriched SVGs: {aml_path}")
    print(f"Saved {condition_b}-enriched SVGs: {wt_path}")

    # ------------------------------------------------------------
    # 画 top SVG barplot
    # ------------------------------------------------------------
    for cond in combined_df["condition"].unique():
        save_fig = os.path.join(outdir, f"top_svg_barplot_{cond}.png")
        plot_top_svg_bar(combined_df, condition=cond, top_n=30, save=save_fig)

    print("Done.")

    return {
        "adata": adata_pp,
        "all_sample_svg": all_sample_df,
        "combined_svg": combined_df,
        "diff_svg": diff_df,
    }

# ============================================================
# 9. 直接运行示例
# ============================================================

if __name__ == "__main__":

    result = run_full_svg_pipeline(
        h5ad_path="/home/zhengyanb/HJ-空转/BMK250409-CR265-ZX01-0101/BMK_DATA_20250804180821_1/03.output/WTkidney_vs_AMLkidney.clustree_result.final.h5ad",
        outdir="/home/zhengyanb/st-analysis/baimaike_result/kidney_WTvsAML/svg_results",

        # 根据你的 adata.obs 实际列名修改
        condition_col="sample",   # 例如 AML / WT 所在列
        sample_col="library_id",         # 每个切片或样本 ID 所在列

        # 根据你的 adata.obsm 实际 key 修改
        spatial_key="spatial",

        condition_a="0",
        condition_b="1",

        # Visium 常用 6；Slide-seq/MERFISH 可用 8~20
        k=6,

        # 测试可用 99/199；正式建议 999
        n_perm=999,

        # SVG 分析通常先在 HVG 上做，速度更合理
        n_top_hvg=3000,

        min_cells_expressed=20,
        seed=123,
    )
