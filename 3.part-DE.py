# 选择关键亚群做分析（PT和TAL）
target_subtypes = ["New PT", "TAL"]
# ===================== 1. 筛选目标亚群：只保留 New PT 和 TAL =====================
adata_subset = kidney_adata[kidney_adata.obs["subtype"].isin(["New PT", "TAL"])].copy()
# 查看一下细胞数量
print(adata_subset.obs[["subtype", "orig.ident"]].value_counts().sort_index())
# ===================== 2. 差异分析：New PT (AML vs WT) =====================
adata_newpt = kidney_adata[kidney_adata.obs["subtype"] == "New PT"].copy()
adata_tal = kidney_adata[kidney_adata.obs["subtype"] == "TAL"].copy()
sc.pp.log1p(adata_newpt)
sc.pp.log1p(adata_tal)
# 做差异基因
# 运行差异分析
sc.tl.rank_genes_groups(
    adata_newpt,
    groupby="orig.ident",
    groups=["AMLkidney"],
    reference="WTkidney",
    method="wilcoxon",
)
sc.tl.rank_genes_groups(
    adata_tal,
    groupby="orig.ident",
    groups=["AMLkidney"],
    reference="WTkidney",
    method="wilcoxon",
)
# 关键修复：必须加 ["AML"]
newpt_de = pd.DataFrame({
    'gene': adata_newpt.uns['rank_genes_groups']['names']["AMLkidney"],
    'log2fc': adata_newpt.uns['rank_genes_groups']['logfoldchanges']["AMLkidney"],
    'pval': adata_newpt.uns['rank_genes_groups']['pvals']["AMLkidney"],
    'padj': adata_newpt.uns['rank_genes_groups']['pvals_adj']["AMLkidney"],
})
tal_de = pd.DataFrame({
    'gene': adata_tal.uns['rank_genes_groups']['names']["AMLkidney"],
    'log2fc': adata_tal.uns['rank_genes_groups']['logfoldchanges']["AMLkidney"],
    'pval': adata_tal.uns['rank_genes_groups']['pvals']["AMLkidney"],
    'padj': adata_tal.uns['rank_genes_groups']['pvals_adj']["AMLkidney"],
})
# 过滤
newpt_de = newpt_de[(newpt_de['padj'] < 0.05) & (newpt_de['log2fc'].abs() > 0.5)]
tal_de = tal_de[(tal_de['padj'] < 0.05) & (tal_de['log2fc'].abs() > 0.5)]
print("New PT 上调：", (newpt_de['log2fc'] > 0).sum())
print("New PT 下调：", (newpt_de['log2fc'] < 0).sum())
print("TAL 上调：", (tal_de['log2fc'] > 0).sum())
print("TAL 下调：", (tal_de['log2fc'] < 0).sum())
# 对差异基因的火山图可视化
def plot_volcano_with_arrow(de_df, title, save_path):
    plt.figure(figsize=(7, 6))
    
    # 背景点
    plt.scatter(de_df['log2fc'], -np.log10(de_df['padj']), 
                c='lightgray', alpha=0.6, s=2)
    
    
    # 显著上下调
    up = de_df[(de_df['padj'] < 0.05) & (de_df['log2fc'] > 0.5)]
    down = de_df[(de_df['padj'] < 0.05) & (de_df['log2fc'] < -0.5)]
    up_5 = de_df[(de_df['padj'] < 0.05) & (de_df['log2fc'] > 0.5)].head(5)
    down_5 = de_df[(de_df['padj'] < 0.05) & (de_df['log2fc'] < -0.5)].tail(6)
    
    plt.scatter(up['log2fc'], -np.log10(up['padj']), 
                c='red', s=4, alpha=0.8, label=f'Upregulated ({len(up)})')
    plt.scatter(down['log2fc'], -np.log10(down['padj']), 
                c='blue', s=4, alpha=0.8, label=f'Downregulated ({len(down)})')
    
    plt.scatter(up_5['log2fc'], -np.log10(up_5['padj']), c='red', s=20, alpha=0.7)
    plt.scatter(down_5['log2fc'], -np.log10(down_5['padj']), c='blue', s=20, alpha=0.7)

    # ===================== 自动画线 + 标注 =====================
    for i, (x, y, g) in enumerate(zip(up_5['log2fc'], -np.log10(up_5['padj']), up_5['gene'])):
        x_txt = x + 6 + i*2
        y_txt = y + i*1
        plt.plot([x, x_txt], [y, y_txt], c='red', linewidth=0.6)  # 引导线
        plt.text(x_txt, y_txt, g, fontsize=8, color='red')

    for i, (x, y, g) in enumerate(zip(down_5['log2fc'], -np.log10(down_5['padj']), down_5['gene'])):
        x_txt = x - 14 - i*3
        y_txt = y + i*1
        plt.plot([x, x_txt], [y, y_txt], c='blue', linewidth=0.6)  # 引导线
        plt.text(x_txt, y_txt, g, fontsize=8, color='blue')

    # 阈值线
    plt.axhline(-np.log10(0.05), ls='--', c='k', lw=0.7)
    plt.axvline(0.5, ls='--', c='k', lw=0.7)
    plt.axvline(-0.5, ls='--', c='k', lw=0.7)

    plt.xlabel('log2FC (AML vs WT)')
    plt.ylabel('-log10(Adjusted p-value)')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
plot_volcano_with_arrow(newpt_de, 'New PT DEGs (AML vs WT)', 'NewPT_volcano.pdf')
plot_volcano_with_arrow(tal_de, 'New PT DEGs (AML vs WT)', 'TAL_volcano.pdf')
