library(flacco)
library(dplyr)
library(rstudioapi)

# set the working directory to the current directory
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

res = data.frame(NULL)
samples <- read.csv("./initial_design_parsed.csv")
for (i in 1:24) {
  for (j in 1:5) {
    for (run in 1:60) {
      tmp_df <- filter(samples, bbob_function == i & bbob_instance == j & seed == run)
      X <- tmp_df[,8:12] 
      y <- tmp_df[,7]
      feat.obj <- createFeatureObject(X = X, y = y, minimize = TRUE, blocks = 2, lower = rep(-5, 5), upper = rep(5, 5))
      fts.ela_distr <- calculateFeatureSet(feat.object = feat.obj, set = "ela_distr")
      fts.ela_meta <- calculateFeatureSet(feat.object = feat.obj, set = "ela_meta")
      fts.disp <- calculateFeatureSet(feat.object = feat.obj, set = "disp")
      fts.nbc <- calculateFeatureSet(feat.object = feat.obj, set = "nbc")
      fts.ic <- calculateFeatureSet(feat.object = feat.obj, set = "ic")
      fts_list <- list(bbob_function = i, bbob_instance = j, seed = run)
      fts_list <- append(fts_list, fts.ela_distr)
      fts_list <- append(fts_list, fts.ela_meta)
      fts_list <- append(fts_list, fts.disp)
      fts_list <- append(fts_list, fts.nbc)
      fts_list <- append(fts_list, fts.ic)
      res <- rbind(res, data.frame(fts_list))
    }
  }
}
new_f = paste0("ELA_features_", basename("./initial_design_parsed.csv"))
write.csv(res, file = new_f)