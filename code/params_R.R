## This script contains some global settings for all scripts in this repo, such as defining the default plotting theme etc.
library(tidyverse)
library(cowplot)
library(ggsci)

theme_cowplot_custom <- theme_cowplot() +
  theme(axis.title = element_text(face="bold"))

theme_set(theme_cowplot_custom)