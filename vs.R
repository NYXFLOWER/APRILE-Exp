# Title     : TODO
# Objective : TODO
# Created by: nyxfer
# Created on: 2020-08-18

library(SysbioTreemaps)
library(dplyr)
library(colorspace)

# pd <- read.csv(file = 'out/tree-map/data.csv')
# head(pd)

df <- read.csv(file = 'Docu/github/PoSePath/out/tree-map/list.csv', row.names = 1)

tm <- voronoiTreemap(
  data = df,
  levels = c("label", "side.effect", "GO"),
  shape = "rectangle",
  error_tol = 0.005,
  maxIteration = 200,
  positioning = "clustered_by_area",
  seed = 1
)
custom_pal_1 <- sequential_hcl(
  n = 20,
  h = c(-46, 78),
  c = c(61, 78, 54),
  l = c(60, 91),
  power = c(0.8, 1),
  rev = TRUE
)
drawTreemap(
  tm,
  color_palette = custom_pal_1,
  color_type = "cell_size",
  color_level = 3,
  label_level = c(1,2, 3),
  label_size = 3,
  label_color = grey(0.5),
  border_color = grey(0.65),
)