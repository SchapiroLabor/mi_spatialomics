## This script contains some global settings for all scripts in this repo, such as defining the default plotting theme etc.
library(tidyverse)
library(cowplot)
library(ggsci)
library(here)
library(cowplot)
library(RColorBrewer)

## Define theme for all ggplots
theme_cowplot_custom <- theme_cowplot() +
  theme(axis.title = element_text(face="bold"))

theme_set(theme_cowplot_custom)

time_palette <- brewer.pal(4, "Dark2")

################
## Seurat analysis for Molkart data
################
create_seurat_sctransform_mcquant <- function(mcquant,sample_ID){
  cell_ids <- mcquant$CellID
  exp_matrix <- mcquant %>%
    select(-c(CellID,X_centroid,Y_centroid,Area,MajorAxisLength,MinorAxisLength,Eccentricity,Solidity,Extent,Orientation))

  cell_features <- mcquant %>%
    select(c(X_centroid,Y_centroid,Area,MajorAxisLength,MinorAxisLength,Eccentricity,Solidity,Extent,Orientation))

  gene_names <- colnames(exp_matrix)
  exp_matrix_t <- t(exp_matrix)
  rownames(exp_matrix_t) <- gene_names
  sample_split <- strsplit(sample_ID, "_")[[1]]
  colnames(exp_matrix_t) <- paste(sample_ID,cell_ids,sep="-cell_")
  metadata <- data.frame("cell_ID" = colnames(exp_matrix_t),
                         "sample_ID" = sample_ID,
                         "timepoint" = sample_split[2],
                         "replicate" = sample_split[3],
                         "slide" = gsub(".mcquant","",sample_split[4]),
                         "X_centroid" = mcquant$X_centroid,
                         "Y_centroid" = mcquant$Y_centroid
  ) %>%
    mutate("Y_centroid" = abs(Y_centroid - max(Y_centroid))) %>% ## Orient cells same as image (y,x)
    select(- cell_ID)
  metadata <- cbind(metadata,cell_features)
  rownames(metadata) <- colnames(exp_matrix_t)

  resolve_object <- CreateSeuratObject(counts = exp_matrix_t,
                                       project = sample,
                                       meta.data = metadata,
                                       min.cells = 10,
                                       min.features = 3)

  return(resolve_object)
}



################
## Deep visual proteomics analysis
################

## Color palette for proteomics analysis
proteome_palette <- c("control" = "cyan4",
                      "MI_remote" = "darkorange",
                      "MI_IZ" =  "purple")

# [1] "#8E0152" "#C51B7D" "#DE77AE" "#F1B6DA" "#FDE0EF" "#F7F7F7"
# [7] "#E6F5D0" "#B8E186" "#7FBC41" "#4D9221" "#276419"


## Plot relative protein abundance for a protein of interest
plot_proteomics_boxplot <- function(norm_table,protein, style = "mean"){

  norm_table <- subset(norm_table,gene == protein)

  if(style == "mean"){
    mean_geom <- stat_summary(
      fun.y = mean,
      geom = "errorbar",
      aes(ymax = ..y.., ymin = ..y..),
      width = 0.3,
      size = 2,
      color = "black")
  }else if(style == "bar"){
    mean_geom <- stat_summary(
      fun.y = mean,
      geom = "bar",
      aes(y = exp,
          color = group),
      width = 0.3,
      size = 1)
  }

  protein_boxplot <- ggplot(norm_table,aes(x = group,y = exp, fill = group)) +
    geom_beeswarm(size =2.5, pch = 21, color = "black", aes(fill = group)) +
    mean_geom +
    labs(x = "Group",
         y = "Normalized protein level",
         title = if_else(protein == "Vwf","vWF",protein)) +
    scale_fill_manual(values = c(proteome_palette[["control"]],
                                 proteome_palette[["MI_remote"]],
                                 proteome_palette[["MI_IZ"]])) +
    scale_color_manual(values = c(proteome_palette[["control"]],
                                 proteome_palette[["MI_remote"]],
                                 proteome_palette[["MI_IZ"]])) +
    theme(legend.position = "none",
          plot.title = element_text(hjust = 0.5))

  return(protein_boxplot)
}

## Function to plot protein expression for different groups
plot_protein_boxplot <- function(){


}

## Function to plot formatted volcano plots from proteomic data
plot_pretty_volcano <- function(de_table,
                                pt_label, pt_size,
                                plot_title,
                                sig_thresh,
                                sig_col = "adj.P.Val",
                                col_pos_logFC = "red",
                                col_neg_logFC = "blue"){

  volcano_plot <- ggplot(data=de_table, aes(x= logFC, y= -log10(pval), label = label_protein)) +
    geom_point(data = subset(de_table, get(sig_col) > sig_thresh),
               size = pt_size, color = "darkgrey") +
    geom_point(data = subset(de_table, get(sig_col) <= sig_thresh  & logFC > 0),
               size = pt_size, color = col_pos_logFC) +
    geom_point(data = subset(de_table, get(sig_col) <= sig_thresh  & logFC < 0),
               size = pt_size, color = col_neg_logFC) +
    labs(title = plot_title)

  return(volcano_plot)
}


## Function by Aurélien Dugroud to read gmt files for decoupler
import_gmt <- function(gmtfile, fast = T)
{
  if(fast)
  {
    genesets = GSEABase::getGmt(con = gmtfile)
    genesets = unlist(genesets)

    gene_to_term =plyr::ldply(genesets,function(geneset){
      temp <- geneIds(geneset)
      temp2 <- setName(geneset)
      temp3 <- as.data.frame(cbind(temp,rep(temp2,length(temp))))

    },.progress = plyr::progress_text())
    names(gene_to_term) <- c("gene","term")
    return(gene_to_term[complete.cases(gene_to_term),])
  }
  else
  {
    genesets = getGmt(con = gmtfile)
    genesets = unlist(genesets)

    gene_to_term <- data.frame(NA,NA)
    names(gene_to_term) <- c("gene","term")
    for (geneset in genesets)
    {
      temp <- geneIds(geneset)
      temp2 <- setName(geneset)
      temp3 <- as.data.frame(cbind(temp,rep(temp2,length(temp))))
      names(temp3) <- c("gene","term")
      gene_to_term <- rbind(gene_to_term,temp3)
    }

    return(gene_to_term[complete.cases(gene_to_term),])
  }
}

## Function for ggplot to summarize df for barplot plotting
data_summary <- function(data, varname, groupnames){
  require(plyr)
  summary_func <- function(x, col){
    c(mean = mean(x[[col]], na.rm=TRUE),
      sd = sd(x[[col]], na.rm=TRUE))
  }
  data_sum<-ddply(data, groupnames, .fun=summary_func,
                  varname)
  data_sum <- rename(data_sum, c("mean" = varname))
  return(data_sum)
}
