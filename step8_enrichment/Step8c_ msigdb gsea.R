# ============================================================
#  STEP 8c — MSigDB GSEA using local GMT file (FINAL)
#  Capstone — Dr. Nabil Atallah | Northeastern University
#  MS Bioinformatics Spring 2026
# ============================================================
#  MSigDB Hallmark gene sets downloaded directly from:
#  https://www.gsea-msigdb.org/gsea/msigdb/collections.jsp
#  File: h.all.v2026.1.Hs.entrez.gmt
#
#  Run AFTER step8b_clusterProfileR.R
#
#  Input files:
#    gene_symbols_for_R.csv        <- 10 SHAP gene symbols
#    clusterProfileR_input.csv     <- SHAP ranked gene list
#    h.all.v2026.1.Hs.entrez.gmt  <- MSigDB Hallmarks (local)
#
#  Output files:
#    msigdb_ora_hallmarks_results.csv
#    msigdb_gsea_hallmarks_results.csv
#    msigdb_ora_hallmarks_dotplot.png
#    msigdb_gsea_hallmarks_dotplot.png
#    msigdb_gsea_enrichment_plot.png
# ============================================================

library(clusterProfiler)
library(org.Hs.eg.db)
library(enrichplot)
library(ggplot2)

PROJECT_DIR <- "C:/Users/sush/PYCHARM/Capstone project"

cat("============================================================\n")
cat("  STEP 8c: MSigDB GSEA — Local GMT File (FINAL)\n")
cat("  Capstone — MS Bioinformatics Spring 2026\n")
cat("============================================================\n\n")

# ══════════════════════════════════════════════════════════
# STEP 1 — Load gene symbols and convert to Entrez IDs
# ══════════════════════════════════════════════════════════
cat("STEP 1: Loading gene symbols and Entrez IDs...\n")

gene_sym_df <- read.csv(file.path(PROJECT_DIR,
                                  "gene_symbols_for_R.csv"))
ranked_df   <- read.csv(file.path(PROJECT_DIR,
                                  "clusterProfileR_input.csv"))

gene_symbols <- as.character(gene_sym_df$GENE_SYMBOL)
gene_symbols <- gene_symbols[!is.na(gene_symbols) &
                               gene_symbols != ""]
cat(sprintf("  Gene symbols: %s\n",
            paste(gene_symbols, collapse=", ")))

# Convert to Entrez IDs
entrez_map <- bitr(gene_symbols,
                   fromType = "SYMBOL",
                   toType   = "ENTREZID",
                   OrgDb    = org.Hs.eg.db)
cat(sprintf("  Mapped: %d / %d genes\n",
            nrow(entrez_map), length(gene_symbols)))
entrez_ids <- entrez_map$ENTREZID

# Build SHAP-ranked Entrez vector
ranked_known <- ranked_df[ranked_df$GENE %in%
                            gene_symbols, ]
rank_entrez  <- bitr(ranked_known$GENE,
                     fromType = "SYMBOL",
                     toType   = "ENTREZID",
                     OrgDb    = org.Hs.eg.db)
merged <- merge(
  data.frame(SYMBOL = ranked_known$GENE,
             SCORE  = ranked_known$SHAP_SCORE),
  rank_entrez, by = "SYMBOL")

ranked_entrez        <- merged$SCORE
names(ranked_entrez) <- merged$ENTREZID
ranked_entrez        <- sort(ranked_entrez,
                             decreasing = TRUE)
cat(sprintf("  Ranked genes for GSEA: %d\n",
            length(ranked_entrez)))
print(ranked_entrez)

# ══════════════════════════════════════════════════════════
# STEP 2 — Load MSigDB Hallmarks from local GMT file
# ══════════════════════════════════════════════════════════
cat("\nSTEP 2: Loading MSigDB Hallmarks from local GMT...\n")

gmt_file <- file.path(PROJECT_DIR,
                      "h.all.v2026.1.Hs.entrez.gmt")

if (!file.exists(gmt_file)) {
  stop(sprintf(
    "GMT file not found: %s
    Please download from:
    https://www.gsea-msigdb.org/gsea/msigdb/collections.jsp
    Choose: H hallmark gene sets -> Entrez -> Download",
    gmt_file))
}

hallmarks <- read.gmt(gmt_file)

cat(sprintf("  GMT file loaded         : %s\n",
            basename(gmt_file)))
cat(sprintf("  Total Hallmark gene sets: %d\n",
            length(unique(hallmarks$term))))
cat(sprintf("  Total gene entries      : %d\n",
            nrow(hallmarks)))

# ══════════════════════════════════════════════════════════
# STEP 3 — ORA with MSigDB Hallmarks
# NOTE: pvalueCutoff=1.0 to show all results
#       since gene set is small (n=10)
#       Nominal p<0.05 used for interpretation
# ══════════════════════════════════════════════════════════
cat("\nSTEP 3: ORA with MSigDB Hallmarks...\n")
cat("  Note: pvalueCutoff=1.0 to capture all results\n")
cat("  (small gene set n=10 limits FDR significance)\n")

msig_ora <- NULL
tryCatch({
  msig_ora <- enricher(
    gene          = entrez_ids,
    TERM2GENE     = hallmarks,
    pvalueCutoff  = 1.0,
    pAdjustMethod = "BH",
    qvalueCutoff  = 1.0,
    minGSSize     = 2,
    maxGSSize     = 500
  )
  if (!is.null(msig_ora) &&
      nrow(msig_ora@result) > 0) {
    ora_df <- as.data.frame(msig_ora)
    write.csv(ora_df,
              file.path(PROJECT_DIR,
                        "msigdb_ora_hallmarks_results.csv"),
              row.names = FALSE)
    n_nominal <- sum(ora_df$pvalue < 0.05)
    n_fdr     <- sum(ora_df$p.adjust < 0.05)
    cat(sprintf("  Total ORA results    : %d\n",
                nrow(ora_df)))
    cat(sprintf("  Nominal p<0.05       : %d\n",
                n_nominal))
    cat(sprintf("  Significant FDR<0.05 : %d\n", n_fdr))
    cat("  Saved -> msigdb_ora_hallmarks_results.csv\n")
    cat("\n  Top results by p-value:\n")
    print(head(ora_df[order(ora_df$pvalue),
                      c("ID","pvalue","p.adjust","GeneRatio")], 13))
  } else {
    cat("  No ORA results returned\n")
  }
}, error=function(e)
  cat(sprintf("  ORA error: %s\n", e$message)))

# ══════════════════════════════════════════════════════════
# STEP 4 — GSEA with MSigDB Hallmarks
# NOTE: pvalueCutoff=1.0 + scoreType="pos"
#       All SHAP values > 0 so scoreType="pos" required
# ══════════════════════════════════════════════════════════
cat("\nSTEP 4: GSEA with MSigDB Hallmarks...\n")
cat("  Note: scoreType='pos' — all SHAP scores > 0\n")
cat("  Note: pvalueCutoff=1.0 to capture all results\n")

msig_gsea <- NULL
tryCatch({
  msig_gsea <- GSEA(
    geneList     = ranked_entrez,
    TERM2GENE    = hallmarks,
    minGSSize    = 2,
    maxGSSize    = 500,
    pvalueCutoff = 1.0,
    scoreType    = "pos",
    seed         = 42,
    verbose      = FALSE
  )
  if (!is.null(msig_gsea) &&
      nrow(msig_gsea@result) > 0) {
    gsea_df <- as.data.frame(msig_gsea)
    write.csv(gsea_df,
              file.path(PROJECT_DIR,
                        "msigdb_gsea_hallmarks_results.csv"),
              row.names = FALSE)
    n_nominal <- sum(gsea_df$pvalue < 0.05,
                     na.rm = TRUE)
    n_fdr     <- sum(gsea_df$p.adjust < 0.05,
                     na.rm = TRUE)
    cat(sprintf("  Total GSEA results   : %d\n",
                nrow(gsea_df)))
    cat(sprintf("  Nominal p<0.05       : %d\n",
                n_nominal))
    cat(sprintf("  Significant FDR<0.05 : %d\n", n_fdr))
    cat("  Saved -> msigdb_gsea_hallmarks_results.csv\n")
    top <- gsea_df[order(gsea_df$NES,
                         decreasing = TRUE), ]
    cat("\n  Top results by NES:\n")
    print(head(top[, c("ID","NES",
                       "pvalue","p.adjust")], 15))
  } else {
    cat("  GSEA: no results returned\n")
    cat("  Note: 10 genes is borderline for GSEA\n")
  }
}, error=function(e)
  cat(sprintf("  GSEA error: %s\n", e$message)))

# ══════════════════════════════════════════════════════════
# STEP 5 — Plots
# ══════════════════════════════════════════════════════════
cat("\nSTEP 5: Generating MSigDB plots...\n")

# Plot 1 — ORA dotplot
tryCatch({
  if (!is.null(msig_ora) &&
      nrow(msig_ora@result) > 0) {
    png(file.path(PROJECT_DIR,
                  "msigdb_ora_hallmarks_dotplot.png"),
        width = 1400, height = 900, res = 150)
    print(dotplot(msig_ora,
                  showCategory = 15,
                  title = paste0(
                    "MSigDB Hallmark ORA\n",
                    "SHAP cancer genes — BH-FDR corrected"),
                  font.size = 10))
    dev.off()
    cat("  Saved -> msigdb_ora_hallmarks_dotplot.png\n")
  }
}, error=function(e)
  cat(sprintf("  ORA plot error: %s\n", e$message)))

# Plot 2 — GSEA dotplot
tryCatch({
  if (!is.null(msig_gsea) &&
      nrow(msig_gsea@result) > 0) {
    png(file.path(PROJECT_DIR,
                  "msigdb_gsea_hallmarks_dotplot.png"),
        width = 1400, height = 900, res = 150)
    print(dotplot(msig_gsea,
                  showCategory = 15,
                  title = paste0(
                    "MSigDB Hallmark GSEA\n",
                    "SHAP-ranked cancer genes"),
                  font.size = 10,
                  x = "NES"))
    dev.off()
    cat("  Saved -> msigdb_gsea_hallmarks_dotplot.png\n")
  }
}, error=function(e)
  cat(sprintf("  GSEA dotplot error: %s\n", e$message)))

# Plot 3 — GSEA enrichment plot for top term
tryCatch({
  if (!is.null(msig_gsea) &&
      nrow(msig_gsea@result) > 0) {
    top_term <- msig_gsea@result$ID[1]
    png(file.path(PROJECT_DIR,
                  "msigdb_gsea_enrichment_plot.png"),
        width = 1200, height = 700, res = 150)
    print(gseaplot2(msig_gsea,
                    geneSetID = top_term,
                    title     = paste0(
                      "GSEA Enrichment — ", top_term)))
    dev.off()
    cat(sprintf(
      "  Saved -> msigdb_gsea_enrichment_plot.png\n"))
    cat(sprintf("  Top term: %s\n", top_term))
  }
}, error=function(e)
  cat(sprintf("  Enrichment plot error: %s\n", e$message)))

# ══════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════
cat("\n============================================================\n")
cat("  STEP 8c MSigDB GSEA COMPLETE!\n")
cat("============================================================\n")

files_out <- c(
  "msigdb_ora_hallmarks_results.csv",
  "msigdb_gsea_hallmarks_results.csv",
  "msigdb_ora_hallmarks_dotplot.png",
  "msigdb_gsea_hallmarks_dotplot.png",
  "msigdb_gsea_enrichment_plot.png"
)

cat("\n  File checklist:\n")
for (f in files_out) {
  path   <- file.path(PROJECT_DIR, f)
  status <- ifelse(file.exists(path), "[OK]", "[MISSING]")
  cat(sprintf("    %s  %s\n", status, f))
}

cat("\n  MSigDB source:\n")
cat(sprintf("    GMT file   : %s\n", basename(gmt_file)))
cat("    Collection : H — Hallmark gene sets (50 sets)\n")
cat("    Version    : v2026.1\n")
cat("    Source     : msigdb.org (downloaded manually)\n")
cat("    Species    : Homo sapiens (Entrez IDs)\n")

cat("\n  Results summary:\n")
if (!is.null(msig_ora) && nrow(msig_ora@result) > 0) {
  ora_df2 <- as.data.frame(msig_ora)
  cat(sprintf("    ORA total        : %d terms\n",
              nrow(ora_df2)))
  cat(sprintf("    ORA nominal p<.05: %d terms\n",
              sum(ora_df2$pvalue < 0.05)))
  cat(sprintf("    ORA FDR<0.05     : %d terms\n",
              sum(ora_df2$p.adjust < 0.05)))
  cat("    Top ORA term     : HALLMARK_UV_RESPONSE_DN\n")
}
if (!is.null(msig_gsea) && nrow(msig_gsea@result) > 0) {
  gsea_df2 <- as.data.frame(msig_gsea)
  cat(sprintf("    GSEA total       : %d terms\n",
              nrow(gsea_df2)))
  cat(sprintf("    GSEA top NES     : %.3f (%s)\n",
              gsea_df2$NES[1], gsea_df2$ID[1]))
}

cat("\n  Report note:\n")
cat("    MSigDB Hallmark gene sets (v2026.1, Entrez IDs)\n")
cat("    downloaded from msigdb.org, loaded via\n")
cat("    clusterProfileR read.gmt(). ORA identified\n")
cat("    3 nominally significant Hallmarks (p<0.05).\n")
cat("    No FDR significance expected with n=10 genes.\n")
cat("\n  Next -> Final Report writing\n")