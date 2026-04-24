# ============================================================
#  STEP 8b — clusterProfileR Pathway Enrichment (FINAL)
#  Capstone — Dr. Nabil Atallah | Northeastern University
#  MS Bioinformatics Spring 2026
# ============================================================
#  Run in RStudio AFTER Step 8 Python script completes
#
#  Input files:
#    clusterProfileR_input.csv   <- SHAP ranked gene list
#    gene_symbols_for_R.csv      <- gene symbols for ORA
#
#  Output files:
#    clusterProfiler_kegg_results.csv
#    clusterProfiler_go_results.csv
#    clusterProfiler_reactome_results.csv
#    clusterProfiler_gsea_kegg.csv
#    clusterProfiler_dotplot_go.png
#    clusterProfiler_barplot_kegg.png
#    clusterProfiler_dotplot_reactome.png
#    clusterProfiler_cnet_kegg.png
#    clusterProfiler_gsea_dotplot.png
#    enrichment_volcano_kegg.png       <- NEW
# ============================================================

# ── Install packages if needed (run ONCE only) ────────────
# if (!requireNamespace("BiocManager", quietly=TRUE))
#   install.packages("BiocManager")
# BiocManager::install("clusterProfiler")
# BiocManager::install("org.Hs.eg.db")
# BiocManager::install("ReactomePA")
# BiocManager::install("enrichplot")
# install.packages("ggplot2")
# install.packages("ggrepel")

library(clusterProfiler)
library(org.Hs.eg.db)
library(ReactomePA)
library(enrichplot)
library(ggplot2)
library(ggrepel)

PROJECT_DIR <- "C:/Users/sush/PYCHARM/Capstone project"

cat("============================================================\n")
cat("  STEP 8b: clusterProfileR Pathway Enrichment\n")
cat("  Capstone — MS Bioinformatics Spring 2026\n")
cat("============================================================\n\n")

# ══════════════════════════════════════════════════════════
# STEP 1 — Load Input Files
# ══════════════════════════════════════════════════════════
cat("STEP 1: Loading input files...\n")

ranked_df   <- read.csv(file.path(PROJECT_DIR,
                                  "clusterProfileR_input.csv"))
gene_sym_df <- read.csv(file.path(PROJECT_DIR,
                                  "gene_symbols_for_R.csv"))

gene_symbols <- as.character(gene_sym_df$GENE_SYMBOL)
gene_symbols <- gene_symbols[!is.na(gene_symbols) &
                               gene_symbols != ""]
cat(sprintf("  Ranked genes : %d\n", nrow(ranked_df)))
cat(sprintf("  Valid symbols: %d\n", length(gene_symbols)))
cat(sprintf("  Genes: %s\n",
            paste(gene_symbols, collapse=", ")))

# ══════════════════════════════════════════════════════════
# STEP 2 — Convert to Entrez IDs
# ══════════════════════════════════════════════════════════
cat("\nSTEP 2: Converting gene symbols to Entrez IDs...\n")

entrez_map <- bitr(gene_symbols,
                   fromType = "SYMBOL",
                   toType   = "ENTREZID",
                   OrgDb    = org.Hs.eg.db)
cat(sprintf("  Mapped: %d / %d genes\n",
            nrow(entrez_map), length(gene_symbols)))
print(entrez_map)
entrez_ids <- entrez_map$ENTREZID

# ══════════════════════════════════════════════════════════
# STEP 3 — KEGG ORA
# ══════════════════════════════════════════════════════════
cat("\nSTEP 3: KEGG Over-Representation Analysis...\n")

kegg_ora <- NULL
tryCatch({
  kegg_ora <- enrichKEGG(
    gene          = entrez_ids,
    organism      = "hsa",
    pvalueCutoff  = 0.05,
    pAdjustMethod = "BH",
    qvalueCutoff  = 0.2)
  if (!is.null(kegg_ora) && nrow(kegg_ora@result) > 0) {
    kegg_df <- as.data.frame(kegg_ora)
    write.csv(kegg_df,
              file.path(PROJECT_DIR,
                        "clusterProfiler_kegg_results.csv"),
              row.names=FALSE)
    cat(sprintf("  Pathways found      : %d\n", nrow(kegg_df)))
    cat(sprintf("  Significant FDR<0.05: %d\n",
                sum(kegg_df$p.adjust < 0.05)))
    cat("  Saved -> clusterProfiler_kegg_results.csv\n")
    print(head(kegg_df[, c("Description","pvalue",
                           "p.adjust","GeneRatio")], 10))
  } else { cat("  No significant KEGG pathways\n") }
}, error=function(e) cat(sprintf("  KEGG error: %s\n",
                                 e$message)))

# ══════════════════════════════════════════════════════════
# STEP 4 — GO Biological Process ORA
# ══════════════════════════════════════════════════════════
cat("\nSTEP 4: GO Biological Process ORA...\n")

go_ora <- NULL
tryCatch({
  go_ora <- enrichGO(
    gene          = entrez_ids,
    OrgDb         = org.Hs.eg.db,
    ont           = "BP",
    pAdjustMethod = "BH",
    pvalueCutoff  = 0.05,
    qvalueCutoff  = 0.2,
    readable      = TRUE)
  if (!is.null(go_ora) && nrow(go_ora@result) > 0) {
    go_df <- as.data.frame(go_ora)
    write.csv(go_df,
              file.path(PROJECT_DIR,
                        "clusterProfiler_go_results.csv"),
              row.names=FALSE)
    cat(sprintf("  GO BP terms found   : %d\n", nrow(go_df)))
    cat(sprintf("  Significant FDR<0.05: %d\n",
                sum(go_df$p.adjust < 0.05)))
    cat("  Saved -> clusterProfiler_go_results.csv\n")
    print(head(go_df[, c("Description","pvalue",
                         "p.adjust","GeneRatio")], 10))
  } else { cat("  No significant GO terms\n") }
}, error=function(e) cat(sprintf("  GO error: %s\n",
                                 e$message)))

# ══════════════════════════════════════════════════════════
# STEP 5 — Reactome ORA
# ══════════════════════════════════════════════════════════
cat("\nSTEP 5: Reactome Pathway ORA...\n")

react_ora <- NULL
tryCatch({
  react_ora <- enrichPathway(
    gene          = entrez_ids,
    pvalueCutoff  = 0.05,
    pAdjustMethod = "BH",
    readable      = TRUE)
  if (!is.null(react_ora) && nrow(react_ora@result) > 0) {
    react_df <- as.data.frame(react_ora)
    write.csv(react_df,
              file.path(PROJECT_DIR,
                        "clusterProfiler_reactome_results.csv"),
              row.names=FALSE)
    cat(sprintf("  Reactome pathways   : %d\n", nrow(react_df)))
    cat(sprintf("  Significant FDR<0.05: %d\n",
                sum(react_df$p.adjust < 0.05)))
    cat("  Saved -> clusterProfiler_reactome_results.csv\n")
    print(head(react_df[, c("Description","pvalue",
                            "p.adjust","GeneRatio")], 10))
  } else { cat("  No significant Reactome pathways\n") }
}, error=function(e) cat(sprintf("  Reactome error: %s\n",
                                 e$message)))

# ══════════════════════════════════════════════════════════
# STEP 6 — GSEA with SHAP-ranked genes
# ══════════════════════════════════════════════════════════
cat("\nSTEP 6: GSEA with SHAP-ranked gene list...\n")

gsea_kegg <- NULL
tryCatch({
  known_symbols <- c("PTEN","NF1","ARID1A","TP53",
                     "ACVR2A","KRAS","EP300","BRAF",
                     "PIK3CA","SMARCA4")
  
  ranked_known <- ranked_df[ranked_df$GENE %in%
                              known_symbols, ]
  cat(sprintf("  Known symbols found: %d\n",
              nrow(ranked_known)))
  
  if (nrow(ranked_known) > 0) {
    rank_entrez <- bitr(
      ranked_known$GENE,
      fromType = "SYMBOL",
      toType   = "ENTREZID",
      OrgDb    = org.Hs.eg.db)
    
    merged <- merge(
      data.frame(SYMBOL = ranked_known$GENE,
                 SCORE  = ranked_known$SHAP_SCORE),
      rank_entrez, by="SYMBOL")
    
    ranked_entrez        <- merged$SCORE
    names(ranked_entrez) <- merged$ENTREZID
    ranked_entrez        <- sort(ranked_entrez,
                                 decreasing=TRUE)
    cat(sprintf("  Final ranked genes : %d\n",
                length(ranked_entrez)))
    
    gsea_kegg <- gseKEGG(
      geneList     = ranked_entrez,
      organism     = "hsa",
      minGSSize    = 3,
      maxGSSize    = 500,
      pvalueCutoff = 1.0,
      scoreType    = "pos",
      verbose      = FALSE,
      seed         = 42)
    
    if (!is.null(gsea_kegg) &&
        nrow(gsea_kegg@result) > 0) {
      gsea_df <- as.data.frame(gsea_kegg)
      write.csv(gsea_df,
                file.path(PROJECT_DIR,
                          "clusterProfiler_gsea_kegg.csv"),
                row.names=FALSE)
      cat(sprintf("  GSEA total results : %d\n",
                  nrow(gsea_df)))
      cat(sprintf("  GSEA significant   : %d\n",
                  sum(gsea_df$p.adjust < 0.05,
                      na.rm=TRUE)))
      cat("  Saved -> clusterProfiler_gsea_kegg.csv\n")
      print(head(gsea_df[, c("Description","NES",
                             "pvalue","p.adjust")], 10))
    } else {
      cat("  GSEA: no results — 10 genes too few\n")
      cat("  ORA results above are sufficient\n")
    }
  }
}, error=function(e) cat(sprintf("  GSEA error: %s\n",
                                 e$message)))

# ══════════════════════════════════════════════════════════
# STEP 7 — Plots
# ══════════════════════════════════════════════════════════
cat("\nSTEP 7: Generating plots...\n")

# Plot 1 — GO dotplot
tryCatch({
  if (!is.null(go_ora) && nrow(go_ora@result) > 0) {
    png(file.path(PROJECT_DIR,
                  "clusterProfiler_dotplot_go.png"),
        width=1400, height=900, res=150)
    print(dotplot(go_ora,
                  showCategory = 15,
                  title = paste0(
                    "GO Biological Process ORA\n",
                    "SHAP cancer genes — BH-FDR corrected"),
                  font.size = 10))
    dev.off()
    cat("  Saved -> clusterProfiler_dotplot_go.png\n")
  }
}, error=function(e) cat(sprintf("  GO plot error: %s\n",
                                 e$message)))

# Plot 2 — KEGG barplot
tryCatch({
  if (!is.null(kegg_ora) && nrow(kegg_ora@result) > 0) {
    png(file.path(PROJECT_DIR,
                  "clusterProfiler_barplot_kegg.png"),
        width=1400, height=900, res=150)
    print(barplot(kegg_ora,
                  showCategory = 15,
                  title = paste0(
                    "KEGG Pathway ORA\n",
                    "SHAP cancer genes — BH-FDR corrected"),
                  font.size = 10))
    dev.off()
    cat("  Saved -> clusterProfiler_barplot_kegg.png\n")
  }
}, error=function(e) cat(sprintf("  KEGG barplot error: %s\n",
                                 e$message)))

# Plot 3 — KEGG Enrichment Volcano plot ← NEW
tryCatch({
  if (!is.null(kegg_ora) && nrow(kegg_ora@result) > 0) {
    kegg_vol <- as.data.frame(kegg_ora)
    kegg_vol$logP        <- -log10(kegg_vol$pvalue)
    kegg_vol$Significant <- ifelse(
      kegg_vol$p.adjust < 0.05, "FDR<0.05", "Not significant")
    kegg_vol$Label <- ifelse(
      kegg_vol$p.adjust < 0.05,
      kegg_vol$Description, "")
    
    p_volcano <- ggplot(kegg_vol,
                        aes(x    = Count,
                            y    = logP,
                            color= Significant,
                            label= Label)) +
      geom_point(size=3, alpha=0.8) +
      geom_hline(yintercept = -log10(0.05),
                 linetype   = "dashed",
                 color      = "red",
                 linewidth  = 0.6) +
      geom_text_repel(size        = 3,
                      max.overlaps= 15,
                      box.padding = 0.4) +
      scale_color_manual(
        values = c("FDR<0.05"       = "#D85A30",
                   "Not significant"= "grey60")) +
      labs(title    = "KEGG Pathway Enrichment Volcano",
           subtitle = "SHAP-ranked cancer genes — BH-FDR corrected",
           x        = "Gene Count",
           y        = expression(-log[10](p-value)),
           color    = "Significance") +
      theme_minimal(base_size=12) +
      theme(plot.title    = element_text(face="bold"),
            legend.position = "bottom")
    
    ggsave(file.path(PROJECT_DIR,
                     "enrichment_volcano_kegg.png"),
           plot   = p_volcano,
           width  = 10,
           height = 7,
           dpi    = 150)
    cat("  Saved -> enrichment_volcano_kegg.png\n")
  }
}, error=function(e) cat(sprintf("  Volcano plot error: %s\n",
                                 e$message)))

# Plot 4 — Reactome dotplot
tryCatch({
  if (!is.null(react_ora) && nrow(react_ora@result) > 0) {
    png(file.path(PROJECT_DIR,
                  "clusterProfiler_dotplot_reactome.png"),
        width=1400, height=900, res=150)
    print(dotplot(react_ora,
                  showCategory = 15,
                  title = paste0(
                    "Reactome Pathway ORA\n",
                    "SHAP cancer genes — BH-FDR corrected"),
                  font.size = 10))
    dev.off()
    cat("  Saved -> clusterProfiler_dotplot_reactome.png\n")
  }
}, error=function(e) cat(sprintf("  Reactome plot error: %s\n",
                                 e$message)))

# Plot 5 — Gene-concept network KEGG
tryCatch({
  if (!is.null(kegg_ora) && nrow(kegg_ora@result) > 0) {
    png(file.path(PROJECT_DIR,
                  "clusterProfiler_cnet_kegg.png"),
        width=1400, height=1000, res=150)
    print(cnetplot(kegg_ora, showCategory=5))
    dev.off()
    cat("  Saved -> clusterProfiler_cnet_kegg.png\n")
  }
}, error=function(e) cat(sprintf("  cnetplot error: %s\n",
                                 e$message)))

# Plot 6 — GSEA dotplot
tryCatch({
  if (!is.null(gsea_kegg) &&
      nrow(gsea_kegg@result) > 0) {
    png(file.path(PROJECT_DIR,
                  "clusterProfiler_gsea_dotplot.png"),
        width=1400, height=900, res=150)
    print(dotplot(gsea_kegg,
                  showCategory = 10,
                  title = "GSEA KEGG — SHAP-ranked genes",
                  font.size = 10))
    dev.off()
    cat("  Saved -> clusterProfiler_gsea_dotplot.png\n")
  }
}, error=function(e) cat(sprintf("  GSEA plot error: %s\n",
                                 e$message)))

# ══════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════
cat("\n============================================================\n")
cat("  STEP 8b clusterProfileR COMPLETE!\n")
cat("============================================================\n")

files_out <- c(
  "clusterProfiler_kegg_results.csv",
  "clusterProfiler_go_results.csv",
  "clusterProfiler_reactome_results.csv",
  "clusterProfiler_gsea_kegg.csv",
  "clusterProfiler_dotplot_go.png",
  "clusterProfiler_barplot_kegg.png",
  "enrichment_volcano_kegg.png",
  "clusterProfiler_dotplot_reactome.png",
  "clusterProfiler_cnet_kegg.png",
  "clusterProfiler_gsea_dotplot.png"
)

cat("\n  File checklist:\n")
for (f in files_out) {
  path   <- file.path(PROJECT_DIR, f)
  status <- ifelse(file.exists(path), "[OK]", "[MISSING]")
  cat(sprintf("    %s  %s\n", status, f))
}

cat("\n  Enrichment summary:\n")
if (!is.null(kegg_ora) && nrow(kegg_ora@result) > 0) {
  cat(sprintf("    KEGG ORA    : %d significant pathways\n",
              sum(as.data.frame(kegg_ora)$p.adjust < 0.05)))
}
if (!is.null(go_ora) && nrow(go_ora@result) > 0) {
  cat(sprintf("    GO BP ORA   : %d significant terms\n",
              sum(as.data.frame(go_ora)$p.adjust < 0.05)))
}
if (!is.null(react_ora) && nrow(react_ora@result) > 0) {
  cat(sprintf("    Reactome ORA: %d significant pathways\n",
              sum(as.data.frame(react_ora)$p.adjust < 0.05)))
}
if (!is.null(gsea_kegg) && nrow(gsea_kegg@result) > 0) {
  cat(sprintf("    GSEA KEGG   : %d significant\n",
              sum(as.data.frame(gsea_kegg)$p.adjust < 0.05,
                  na.rm=TRUE)))
} else {
  cat("    GSEA KEGG   : insufficient genes — ORA sufficient\n")
}

cat("\n  Next -> step8c_msigdb_gsea.R\n")
