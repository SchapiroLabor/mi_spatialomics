library(tidyverse)
library(mistyR)
library(rlist)
library(FNN)
library(future)
library(cowplot)
library(igraph)
library(ClusterR)

plan(multisession, workers = 8)


# MISTy ----

all.data <- read_tsv("molcart.misty_celltype_table.tsv")

samples <- all.data %>%
  pull(sample_ID) %>%
  unique()
cts <- all.data %>%
  pull(anno_cell_type_lv2) %>%
  unique()

cts.names <- make.names(cts, allow_ = FALSE)


samples %>% walk(\(sample){
  
  sample.cells <- all.data %>%
    filter(sample_ID == sample) %>%
    pull(anno_cell_type_lv2) %>%
    map(~ .x == cts) %>%
    list.rbind() %>%
    `colnames<-`(cts.names) %>%
    as_tibble()

  sample.pos <- all.data %>%
    filter(sample_ID == sample) %>%
    select(X_centroid, Y_centroid)

  #l <- ceiling(knn.dist(sample.pos, 1) %>% mean())
  #l <- 20/0.138
  #l <- 50/0.138
  l <- 100/0.138

  misty.views.cts <- create_initial_view(sample.cells) %>%
    add_paraview(sample.pos, l) %>%
    rename_view(paste0("paraview.", l), "paraview") %>%
    select_markers("intraview", where(~ sd(.) != 0))
  
  run_misty(misty.views.cts, sample, "results_cts_100.sqm", bypass.intra = TRUE)
})


# Output ----

interaction_communities_info <- function(misty.results, concat.views, view,
                                         trim = 0, trim.measure = "gain.R2", 
                                         cutoff = 1, resolution = 1) {
  
  
  inv <- sign((stringr::str_detect(trim.measure, "gain") |
                 stringr::str_detect(trim.measure, "RMSE", negate = TRUE)) - 0.5)
  
  targets <- misty.results$improvements.stats %>%
    dplyr::filter(
      measure == trim.measure,
      inv * mean >= inv * trim
    ) %>%
    dplyr::pull(target)
  
  view.wide <- misty.results$importances.aggregated %>%
    filter(view == !!view) %>%
    pivot_wider(
      names_from = "Target", values_from = "Importance",
      id_cols = -c(view, nsamples)
    ) %>% mutate(across(-c(Predictor,all_of(targets)), \(x) x = NA))
  
  
  mistarget <- setdiff(view.wide$Predictor, colnames(view.wide)[-1])
  mispred <- setdiff(colnames(view.wide)[-1], view.wide$Predictor)
  
  if(length(mispred) != 0){
    view.wide.aug <- view.wide %>% add_row(Predictor = mispred)
  } else {
    view.wide.aug <- view.wide
  }
  
  if(length(mistarget) != 0){
    view.wide.aug <- view.wide.aug %>% 
      bind_cols(mistarget %>% 
                  map_dfc(~tibble(!!.x := NA)))
  }
  
  A <- view.wide.aug %>%
    column_to_rownames("Predictor") %>%
    as.matrix()
  A[A < cutoff | is.na(A)] <- 0
  
  ## !!! Was buggy
  G <- graph.adjacency(A[,rownames(A)], mode = "plus", weighted = TRUE) %>%
    set.vertex.attribute("name", value = names(V(.))) %>%
    delete.vertices(which(degree(.) == 0))
  
  Gdir <- graph.adjacency(A[,rownames(A)], "directed", weighted = TRUE) %>%
    set.vertex.attribute("name", value = names(V(.))) %>%
    delete.vertices(which(degree(.) == 0))
  
  C <- cluster_leiden(G, resolution_parameter = resolution, n_iterations = -1)
  
  mem <- membership(C)
  
  Gdir <- set_vertex_attr(Gdir, "community", names(mem), as.numeric(mem))
  
  # careful here the first argument is the predictor and the second the target, 
  # it might need to come from different view
  corrs <- as_edgelist(Gdir) %>% apply(1, \(x) cor(
    concat.views[[view]][, x[1]],
    concat.views[["intraview"]][, x[2]]
  )) %>% replace_na(0)
  
  Gdir <- set_edge_attr(Gdir, "cor", value = corrs)
  return(Gdir)
}

groups <- samples %>% str_extract("(?<=sample_).+(?=_r)") %>% unique()

misty.results.g <- groups %>% map(~ collect_results("results_cts_100.sqm", .x))
names(misty.results.g) <- groups

dir.create("misty_figures_100")

misty.results.g %>% iwalk(\(misty.results, cond){
  plot.list <- list()
  plot_improvement_stats(misty.results, "gain.R2")
  plot.list <- list.append(plot.list, last_plot())
  plot_interaction_heatmap(misty.results, "paraview", cutoff = 0.5, clean = TRUE, trim = 5)
  plot.list <- list.append(plot.list, last_plot())
  plot_grid(plotlist = plot.list, ncol = 2)
  ggsave(paste0("misty_figures_100/", cond, "_stats.pdf"), width = 10, height = 10)
})

dir.create("graphs", recursive = TRUE)


names(misty.results.g) %>% walk(\(g){
  concat.views <- samples %>% str_subset(g) %>% map(\(sample){
    sample.cells <- all.data %>%
      filter(sample_ID == sample) %>%
      pull(anno_cell_type_lv2) %>%
      map(~ .x == cts) %>%
      list.rbind() %>%
      `colnames<-`(cts.names) %>%
      as_tibble()
    
    sample.pos <- all.data %>%
      filter(sample_ID == sample) %>%
      select(X_centroid, Y_centroid)
    
    #l <- ceiling(knn.dist(sample.pos, 1) %>% mean())
    #l <- 20/0.138
    #l <- 50/0.138
    l <- 100/0.138
    
    misty.views.cts <- create_initial_view(sample.cells) %>%
      add_paraview(sample.pos, l) %>%
      rename_view(paste0("paraview.", l), "paraview") %>%
      select_markers("intraview", where(~ sd(.) != 0))
    
  }) %>% list.clean() %>% reduce(\(acc, nxt){
    list(
      intraview = bind_rows(acc[["intraview"]], nxt[["intraview"]]),
      paraview = bind_rows(acc[["paraview"]], nxt[["paraview"]])
    )
  }, .init = list(intraview = NULL, paraview = NULL))
  
  
  out <- interaction_communities_info(misty.results.g[[g]], concat.views, 
                                      view = "paraview", trim = 5, cutoff = 0.5)
  write_graph(out, paste0("graphs/para.100.", g, ".graphml"), "graphml")
})


cellular_neighborhoods <- function(sample.cells, sample.pos, n, k){
  misty.views  <- create_initial_view(sample.cells) %>% add_paraview(sample.pos, family = "constant", l = n)
  clust <- KMeans_rcpp(misty.views[[paste0("paraview.",n)]], k)
  return(clust$clusters) 
}
