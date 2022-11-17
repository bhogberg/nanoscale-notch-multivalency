library(data.table)
library(magrittr)
library(tximport)
library(DESeq2)

## differential expression
# transcript to gene map
t2g <- fread("gencode.v38.metadata.HGNC.gz", header = F)

# quantification files
fls.notch <- list.files("quant", "quant.sf", recursive = T, full.names = T)
names(fls.notch) <- lapply(strsplit(fls.notch, "/"), "[[", 2)

meta.notch <- data.table(names(fls.notch))[,tstrsplit(V1, "_")]
colnames(meta.notch) <- c("project", "condition", "replicate")

# load quants
txi.notch <- tximport(fls.notch, "salmon", tx2gene = t2g[,1:2]) # abundance and counts from this object uploaded as processed data to ArrayExpress

# differential expression based on 'condition'
dds.notch <- DESeqDataSetFromTximport(txi.notch, meta.notch, ~condition)
dds.notch %<>% DESeq()

# results vs. empty
res.mono <- as.data.table(as.data.frame(results(dds.notch, contrast = c("condition", "monomer", "empty"))), keep.rownames = T)
res.sat <- as.data.table(as.data.frame(results(dds.notch, contrast = c("condition", "saturated", "empty"))), keep.rownames = T)

sig.genes <- unique(c(res.mono[padj < 0.05 & abs(log2FoldChange) > 0.5, rn], res.sat[padj < 0.05 & abs(log2FoldChange) > 0.5, rn]))

# format expression from TPM table
tpm.melt.notch <- as.data.table(melt(txi.notch$abundance))
tpm.melt.notch[, avg := mean(value), by="Var1"]
tpm.melt.notch[, z := scale(value), by="Var1"]
tpm.melt.notch[, c("project", "condition", "replicate") := tstrsplit(Var2, "_")]

## PLOT
library(ggplot2)
library(cowplot)
library(ggrepel)
# volcano plots
p.volcano.mono <-
ggplot(res.mono, aes(y=-log10(padj), x=log2FoldChange, col=rank(padj) <= 20 )) + 
  geom_point(show.legend = F) +
  geom_text_repel(data=res.mono[rank(padj) <= 20], show.legend = F, aes(label=rn)) +
  coord_cartesian(xlim=c(-4,4), ylim=c(0,82)) +
  scale_color_manual(values=c("black", "red")) +
  ggtitle("Monomer vs Empty") +
  theme_cowplot()

p.volcano.sat <-
ggplot(res.sat, aes(y=-log10(padj), x=log2FoldChange, col=rank(padj) <= 20 )) + 
  geom_point(show.legend = F) +
  geom_text_repel(data=res.sat[rank(padj) <= 20], show.legend = F, aes(label=rn)) +
  coord_cartesian(xlim=c(-4,4), ylim=c(0,82)) +
  scale_color_manual(values=c("black", "red")) +
  ggtitle("Saturated vs Empty") +
  theme_cowplot()

# Expression box plots
p.boxplot.exprs <-
ggplot(tpm.melt.notch[Var1 %in% sig.genes], aes(x=condition, y=value, col=condition)) +
  geom_boxplot(outlier.shape = NA) +
  geom_point() +
  facet_wrap(~Var1, scales="free_y") +
  scale_color_brewer(palette="Set1") +
  labs(y="Expression (TPM)") +
  expand_limits(y=0) +
  theme_cowplot() +
  theme(strip.background = element_blank(), panel.border = element_rect(colour = "black"), axis.title.x = element_blank(), axis.text.x = element_blank() )

p.boxplot.exprs2 <-
ggplot(tpm.melt.notch[Var1 %in% c("SOX1", "SOX2", "SOX3", "HES5", "NRARP", "NOTCH1", "NOTCH2", "NOTCH3", "NOTCH4", "JAG1", "DLL1", "DLL4", "LFNG", "MFNG", "RFNG", "OTX2", "FOXG1", "ITGB3", "ITGB5", "ITGA5", "ITGB6", "ITGB8", "ITGA8", "ITGA2B", "ITGB1", "CD36")], aes(x=condition, y=value, col=condition)) +
  geom_boxplot(outlier.shape = NA) +
  geom_point() +
  facet_wrap(~Var1, scales="free_y") +
  scale_color_brewer(palette="Set1") +
  labs(y="Expression (TPM)") +
  expand_limits(y=c(0,3)) +
  theme_cowplot() +
  theme(strip.background = element_blank(), panel.border = element_rect(colour = "black"), axis.title.x = element_blank(), axis.text.x = element_blank() )

# Heatmap
library(ComplexHeatmap)
library(circlize)
library(RColorBrewer)
mat.sig.notch <- as.matrix(data.frame(dcast(tpm.melt.notch[Var1 %in% sig.genes], Var1~Var2, value.var = "z"), row.names = 1))

p.heat.siggenes <-
Heatmap(
  matrix = mat.sig.notch, 
  name = "Z-score", 
  border = "black",
  col = colorRamp2(seq(-2,2, length.out = 11), rev(brewer.pal(11, "RdBu"))),
  use_raster = T
)

dat.gs <- fread("genesets/GO0030100_regulation_of_endocytosis.txt", header = F) # GO:0030100 Regulation of Endocytosis for homo sapiens from the Gene Ontology release 2022-11-03, columns are: gene_name, go_id, go_name, organism
dat.gs[, n := .N, by="V3"]

library(viridisLite)
mat.gs.notch <- na.omit(as.matrix(data.frame(dcast(tpm.melt.notch[Var1 %in% dat.gs$V1 & avg > 1], Var1~Var2, value.var = "value"), row.names = 1)))

rha.gs1 <- rowAnnotation("Geneset" = sapply(unique(dat.gs[n>10,V3]), function(x) row.names(mat.gs.notch) %in% dat.gs[V3 == x, V1]), col= list("Geneset" = setNames(grey.colors(2), c(TRUE, FALSE)) ), simple_anno_size= unit(0.3, "cm"), border = T )
rha.gs2 <- rowAnnotation("Variance" = anno_barplot(rowVars(log2(mat.gs.notch+1)), ylim=c(0,1), gp = gpar(fill = "black")))

p.heat.geneset <-
Heatmap(
  matrix = log2(mat.gs.notch +1), 
  name = "log2(TPM+1)", 
  border = "black",
  col = colorRamp2(seq(0, 10, length.out = 11), viridis(11)),
  left_annotation = rha.gs1,
  right_annotation = rha.gs2, 
  show_row_names = F,
  use_raster = T
)

## Export
# tables
fwrite(res.mono[order(padj, log2FoldChange)], "data/DiffExprs_MonomerVsEmpty.tsv", quote = F, sep = "\t")
fwrite(res.sat[order(padj, log2FoldChange)], "data/DiffExprs_SaturatedVsEmpty.tsv", quote = F, sep = "\t")

# figures
ggsave2("plots/volcano.pdf", height = 4, width = 8, plot_grid(p.volcano.mono, p.volcano.sat))

ggsave2("plots/expression.pdf", height = 12, width = 12, p.boxplot.exprs)
ggsave2("plots/expression2.pdf", height = 7, width = 8, p.boxplot.exprs2)

pdf("plots/heatmap_siggenes.pdf", width = 5, height = 12)
  p.heat.siggenes
dev.off()

pdf("plots/heatmap_geneset.pdf", width = 5, height = 8)
  p.heat.geneset
dev.off()