# ==============================================================================
#  NETFLIX CONTENT CLUSTERING & ANALYSIS USING R
#  Mini Project  |  Unsupervised Learning (K-Means, Hierarchical, PCA)
#  Dataset : Kaggle - "Netflix Movies and TV Shows" (netflix_titles.csv)
# ------------------------------------------------------------------------------
#  HOW TO RUN
#   1. Put this file and netflix_titles.csv in the SAME folder.
#   2. In RStudio:  Session > Set Working Directory > To Source File Location
#   3. Click "Source"  (or run line by line using Ctrl + Enter)
#  Charts are saved in /charts, tables in /results, clean data in /data
# ==============================================================================

# ------------------------------------------------------------------------------
# STAGE 0 : SETUP
# ------------------------------------------------------------------------------
# install.packages(c("dplyr","tidyr","stringr","ggplot2","cluster",
#                    "corrplot","randomForest","caret","shiny"))
suppressPackageStartupMessages({
  library(dplyr);   library(tidyr);  library(stringr); library(ggplot2)
  library(cluster); library(corrplot); library(randomForest); library(caret)
})

set.seed(123)                                   # reproducible results
dir.create("charts",  showWarnings = FALSE)
dir.create("results", showWarnings = FALSE)
dir.create("data",    showWarnings = FALSE)
sink("results/console_output.txt", split = TRUE) # save console output to a file

theme_set(theme_minimal(base_size = 13))
nf_red  <- "#E50914"; nf_dark <- "#221F1F"; nf_grey <- "#B3B3B3"
save_plot <- function(p, name, w = 9, h = 5.5)
  ggsave(file.path("charts", name), p, width = w, height = h, dpi = 150, bg = "white")

# ------------------------------------------------------------------------------
# STAGE 1 : LOADING THE DATASET
# ------------------------------------------------------------------------------
netflix <- read.csv("netflix_titles.csv", stringsAsFactors = FALSE,
                    na.strings = c("", "NA"), fileEncoding = "UTF-8")
cat("\n---- head ----\n");     print(head(netflix, 3))
cat("\n---- dim ----\n");      print(dim(netflix))
cat("\n---- names ----\n");    print(names(netflix))
cat("\n---- str ----\n");      str(netflix)
cat("\n---- summary ----\n");  print(summary(netflix))
n_raw <- nrow(netflix)

# ------------------------------------------------------------------------------
# STAGE 2 : DATA CLEANING
# ------------------------------------------------------------------------------
# 2.1 Missing values
na_summary <- colSums(is.na(netflix))
cat("\n---- Missing values per column ----\n"); print(na_summary)
na_pct <- round(100 * na_summary / nrow(netflix), 2)
write.csv(data.frame(column = names(na_summary), missing = as.integer(na_summary),
                     percent = as.numeric(na_pct)),
          "results/missing_values.csv", row.names = FALSE)

# 2.2 Duplicates
dup_id    <- sum(duplicated(netflix$show_id))
dup_title <- sum(duplicated(netflix[, c("title", "type", "release_year")]))
cat("\nDuplicate show_id rows       :", dup_id,
    "\nDuplicate title/type/year rows:", dup_title, "\n")
netflix <- netflix[!duplicated(netflix[, c("title", "type", "release_year")]), ]

# 2.3 Handle missing values
#     director / cast / country  -> "Unknown"  (too many missing to drop)
#     date_added / rating / duration -> only a handful missing, so drop rows
netflix$director[is.na(netflix$director)] <- "Unknown"
netflix$cast[is.na(netflix$cast)]         <- "Unknown"
netflix$country[is.na(netflix$country)]   <- "Unknown"
n_before <- nrow(netflix)
netflix  <- netflix %>% filter(!is.na(date_added), !is.na(rating), !is.na(duration))
cat("Rows dropped (missing date/rating/duration):", n_before - nrow(netflix), "\n")

# 2.4 Convert text columns to numeric features (feature engineering)
netflix <- netflix %>% mutate(
  date_added_d = as.Date(trimws(date_added), format = "%B %d, %Y"),
  year_added   = as.integer(format(date_added_d, "%Y")),
  month_added  = as.integer(format(date_added_d, "%m")),
  duration_num = as.numeric(str_extract(duration, "[0-9]+")),     # min or seasons
  main_country = str_trim(str_split_fixed(country,  ",", 2)[, 1]),
  main_genre   = str_trim(str_split_fixed(listed_in, ",", 2)[, 1]),
  n_genres     = str_count(listed_in, ",") + 1,
  n_cast       = ifelse(cast == "Unknown", 0, str_count(cast, ",") + 1),
  has_director = as.integer(director != "Unknown"),
  content_age  = pmax(year_added - release_year, 0),               # years before it reached Netflix
  is_movie     = as.integer(type == "Movie"),
  maturity     = case_when(
    rating %in% c("TV-Y", "TV-Y7", "TV-Y7-FV", "G", "TV-G") ~ 1,   # Kids
    rating %in% c("PG", "TV-PG")                            ~ 2,   # Older kids
    rating %in% c("PG-13", "TV-14")                         ~ 3,   # Teens
    rating %in% c("R", "TV-MA", "NC-17")                    ~ 4,   # Adults
    TRUE ~ NA_real_),                                              # NR / UR / unknown
  maturity_label = factor(maturity, levels = 1:4,
                          labels = c("Kids", "Older Kids", "Teens", "Adults"))
)
n_before <- nrow(netflix)
netflix  <- netflix %>% filter(!is.na(maturity), !is.na(duration_num), !is.na(year_added))
cat("Rows dropped (unrated NR/UR or bad duration):", n_before - nrow(netflix), "\n")
cat("Rows after cleaning:", nrow(netflix), "\n")

# 2.5 Genre dummy variables (top-10 genres, exact match on each genre tag)
genre_list   <- strsplit(netflix$listed_in, ",\\s*")
top_genres   <- names(sort(table(unlist(genre_list)), decreasing = TRUE))[1:10]
genre_flags  <- sapply(top_genres, function(g)
                  as.integer(vapply(genre_list, function(x) g %in% x, logical(1))))
colnames(genre_flags) <- paste0("g_", make.names(top_genres))
genre_lookup <- setNames(top_genres, colnames(genre_flags))   # g_ column -> readable genre name
netflix <- bind_cols(netflix, as.data.frame(genre_flags))
cat("\nTop-10 genre tags used as features:\n"); print(top_genres)

# 2.6 Outlier treatment  (see boxplots in Stage 3)
#     Movie minutes and TV seasons are on different scales, so duration is
#     standardised WITHIN each type and then capped at +/- 3 SD (winsorising).
netflix <- netflix %>% group_by(type) %>%
  mutate(duration_z = pmin(pmax(as.numeric(scale(duration_num)), -3), 3)) %>% ungroup() %>%
  mutate(content_age_log = log1p(content_age)) %>% as.data.frame()

cat("\n---- Cleaned data structure ----\n"); str(netflix[, 1:12])

# ------------------------------------------------------------------------------
# STAGE 3 : DATA VISUALIZATION (EDA)
# ------------------------------------------------------------------------------
type_cols <- c("Movie" = nf_red, "TV Show" = nf_dark)

# 3.1 Movies vs TV Shows
d <- netflix %>% count(type) %>% mutate(pct = round(100 * n / sum(n), 1))
p <- ggplot(d, aes(type, n, fill = type)) + geom_col(width = 0.6) +
  geom_text(aes(label = paste0(n, " (", pct, "%)")), vjust = -0.5, fontface = "bold") +
  scale_fill_manual(values = type_cols) + ylim(0, max(d$n) * 1.12) +
  labs(title = "Movies vs TV Shows on Netflix", x = NULL, y = "Count") +
  theme(legend.position = "none")
save_plot(p, "01_type_count.png", 7, 5)

# 3.2 Release year trend
p <- netflix %>% filter(release_year >= 2000) %>% count(release_year, type) %>%
  ggplot(aes(release_year, n, colour = type)) + geom_line(linewidth = 1.2) +
  geom_point(size = 1.8) + scale_colour_manual(values = type_cols) +
  labs(title = "Titles by Release Year (2000 onwards)", x = "Release year",
       y = "Number of titles", colour = NULL) + theme(legend.position = "top")
save_plot(p, "02_release_year_trend.png")

# 3.3 Content added to Netflix per year
p <- netflix %>% count(year_added, type) %>%
  ggplot(aes(year_added, n, fill = type)) + geom_col() +
  scale_fill_manual(values = type_cols) +
  labs(title = "Content Added to Netflix per Year", x = "Year added", y = "Titles", fill = NULL) +
  theme(legend.position = "top")
save_plot(p, "03_content_added_per_year.png")

# 3.4 Top 10 countries
p <- netflix %>% filter(main_country != "Unknown") %>% count(main_country, sort = TRUE) %>%
  slice_head(n = 10) %>%
  ggplot(aes(reorder(main_country, n), n)) + geom_col(fill = nf_red) +
  geom_text(aes(label = n), hjust = -0.2) + coord_flip() +
  scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
  labs(title = "Top 10 Content-Producing Countries", x = NULL, y = "Titles")
save_plot(p, "04_top_countries.png")

# 3.5 Top 10 genres
p <- data.frame(table(unlist(genre_list))) %>% arrange(desc(Freq)) %>% slice_head(n = 10) %>%
  ggplot(aes(reorder(Var1, Freq), Freq)) + geom_col(fill = nf_dark) +
  geom_text(aes(label = Freq), hjust = -0.2) + coord_flip() +
  scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
  labs(title = "Top 10 Genres", x = NULL, y = "Titles")
save_plot(p, "05_top_genres.png")

# 3.6 Rating distribution
p <- netflix %>% count(rating, type) %>%
  ggplot(aes(reorder(rating, -n), n, fill = type)) + geom_col() +
  scale_fill_manual(values = type_cols) +
  labs(title = "Rating Distribution", x = "Rating", y = "Titles", fill = NULL) +
  theme(legend.position = "top")
save_plot(p, "06_rating_distribution.png")

# 3.7 Duration: movie minutes and TV seasons
p1 <- netflix %>% filter(type == "Movie") %>%
  ggplot(aes(duration_num)) + geom_histogram(bins = 40, fill = nf_red, colour = "white") +
  labs(title = "Movie Duration (minutes)", x = "Minutes", y = "Count")
p2 <- netflix %>% filter(type == "TV Show") %>% count(duration_num) %>%
  ggplot(aes(factor(duration_num), n)) + geom_col(fill = nf_dark) +
  labs(title = "TV Show Length (seasons)", x = "Seasons", y = "Count")
save_plot(p1, "07a_movie_duration.png", 7, 4.5)
save_plot(p2, "07b_tv_seasons.png",     7, 4.5)

# 3.8 Maturity level by type
p <- netflix %>% count(maturity_label, type) %>%
  ggplot(aes(maturity_label, n, fill = type)) + geom_col(position = "dodge") +
  scale_fill_manual(values = type_cols) +
  labs(title = "Audience Maturity Level", x = NULL, y = "Titles", fill = NULL) +
  theme(legend.position = "top")
save_plot(p, "08_maturity_level.png", 8, 5)

# 3.9 Correlation matrix of numeric features
num_cols <- c("release_year", "year_added", "content_age", "duration_num", "n_genres",
              "n_cast", "has_director", "maturity", "is_movie")
cor_mat <- cor(netflix[, num_cols], use = "pairwise.complete.obs")
cat("\n---- Correlation matrix ----\n"); print(round(cor_mat, 2))
png("charts/09_correlation_matrix.png", width = 1100, height = 1000, res = 150)
corrplot(cor_mat, method = "color", type = "upper", addCoef.col = "black",
         number.cex = 0.65, tl.cex = 0.8, tl.col = "black", tl.srt = 45,
         col = colorRampPalette(c("#221F1F", "white", "#E50914"))(200),
         title = "Correlation Matrix of Numeric Features", mar = c(0, 0, 2, 0))
dev.off()

# 3.10 Boxplots for outlier checking
box_df <- netflix %>%
  mutate(movie_minutes = ifelse(type == "Movie", duration_num, NA)) %>%
  select(release_year, content_age, movie_minutes, n_cast, n_genres) %>%
  pivot_longer(everything(), names_to = "var", values_to = "val")
p <- ggplot(box_df, aes(x = "", y = val)) +
  geom_boxplot(fill = nf_grey, outlier.colour = nf_red, outlier.alpha = 0.5, na.rm = TRUE) +
  facet_wrap(~var, scales = "free", nrow = 1) +
  labs(title = "Boxplots for Checking Outliers", x = NULL, y = "Value")
save_plot(p, "10_boxplots_outliers.png", 11, 5)

# ------------------------------------------------------------------------------
# STAGE 4 : FEATURE SELECTION & SCALING FOR CLUSTERING
# ------------------------------------------------------------------------------
feature_cols <- c("is_movie", "duration_z", "release_year", "content_age_log",
                  "n_genres", "n_cast", "has_director", "maturity", "year_added",
                  grep("^g_", names(netflix), value = TRUE))
X <- scale(netflix[, feature_cols])           # z-score scaling (mean 0, sd 1)
cat("\nClustering matrix:", nrow(X), "rows x", ncol(X), "features\n")

# ------------------------------------------------------------------------------
# STAGE 5 : MODEL 1 - K-MEANS CLUSTERING
# ------------------------------------------------------------------------------
# 5.1 Elbow method (total within-cluster sum of squares)
ks  <- 2:10
wss <- sapply(ks, function(k) kmeans(X, centers = k, nstart = 20, iter.max = 50)$tot.withinss)
p <- ggplot(data.frame(k = ks, wss = wss), aes(k, wss)) +
  geom_line(colour = nf_red, linewidth = 1.2) + geom_point(size = 3, colour = nf_dark) +
  scale_x_continuous(breaks = ks) +
  labs(title = "Elbow Method", x = "Number of clusters (k)", y = "Total within-cluster SS")
save_plot(p, "11_elbow_method.png", 8, 5)

# 5.2 Silhouette score (computed on a 3000-row random sample for speed)
idx <- sample(nrow(X), 3000)
d_s <- dist(X[idx, ])
sil <- sapply(ks, function(k) {
  km <- kmeans(X[idx, ], centers = k, nstart = 20, iter.max = 50)
  mean(silhouette(km$cluster, d_s)[, 3])
})
p <- ggplot(data.frame(k = ks, sil = sil), aes(k, sil)) +
  geom_line(colour = nf_red, linewidth = 1.2) + geom_point(size = 3, colour = nf_dark) +
  scale_x_continuous(breaks = ks) +
  labs(title = "Average Silhouette Width", x = "Number of clusters (k)", y = "Silhouette score")
save_plot(p, "12_silhouette_scores.png", 8, 5)
cat("\nSilhouette scores:\n"); print(round(setNames(sil, ks), 4))

# Silhouette keeps rising slowly (no sharp peak) and the elbow bends around k = 5-7.
# k = 2 only separates Movies from TV Shows, and k >= 8 splits clusters into tiny,
# hard-to-explain groups. k = 6 gives clean, interpretable segments -> chosen.
k_final <- 6
cat("Chosen number of clusters (k):", k_final,
    "| silhouette at k = 6:", round(sil[ks == 6], 4), "\n")

# 5.3 Final K-Means
km <- kmeans(X, centers = k_final, nstart = 50, iter.max = 100)
netflix$cluster <- km$cluster
cat("\nCluster sizes:\n"); print(table(netflix$cluster))
cat("Between-SS / Total-SS:", round(100 * km$betweenss / km$totss, 2), "%\n")

# ------------------------------------------------------------------------------
# STAGE 6 : MODEL 2 - HIERARCHICAL CLUSTERING (Ward's method)
# ------------------------------------------------------------------------------
idx_h <- sample(nrow(X), 1500)
d_h   <- dist(X[idx_h, ])
hc    <- hclust(d_h, method = "ward.D2")
png("charts/13_dendrogram.png", width = 1400, height = 800, res = 150)
plot(hc, labels = FALSE, hang = -1, main = "Hierarchical Clustering Dendrogram (1500-title sample)",
     xlab = "", sub = "Ward.D2 linkage")
rect.hclust(hc, k = k_final, border = c("#E50914", "#221F1F", "#4F8EF7", "#F2B134", "#2CA58D", "#9B59B6")[1:k_final])
dev.off()
hc_cl  <- cutree(hc, k = k_final)
sil_hc <- mean(silhouette(hc_cl, d_h)[, 3])
km_h   <- kmeans(X[idx_h, ], centers = k_final, nstart = 50)
sil_km <- mean(silhouette(km_h$cluster, d_h)[, 3])
cat("\nSilhouette (same 1500 sample) -> K-Means:", round(sil_km, 4),
    "| Hierarchical:", round(sil_hc, 4), "\n")

# ------------------------------------------------------------------------------
# STAGE 7 : PCA (visualising clusters in 2-D)
# ------------------------------------------------------------------------------
pca     <- prcomp(X)
var_exp <- 100 * pca$sdev^2 / sum(pca$sdev^2)
cat("\nVariance explained by PC1, PC2, PC3:", round(var_exp[1:3], 2), "\n")
cat("Cumulative variance PC1-PC2:", round(sum(var_exp[1:2]), 2), "%\n")
netflix$PC1 <- pca$x[, 1]; netflix$PC2 <- pca$x[, 2]

p <- ggplot(data.frame(PC = 1:10, var = var_exp[1:10]), aes(PC, var)) +
  geom_col(fill = nf_dark) + geom_line(colour = nf_red, linewidth = 1) + geom_point(colour = nf_red, size = 2.5) +
  scale_x_continuous(breaks = 1:10) +
  labs(title = "PCA Scree Plot", x = "Principal component", y = "% variance explained")
save_plot(p, "14_pca_scree.png", 8, 5)

# ------------------------------------------------------------------------------
# STAGE 8 : CLUSTER PROFILING
# ------------------------------------------------------------------------------
mode_of <- function(x) names(sort(table(x), decreasing = TRUE))[1]
top_tags <- function(x, n = 2)               # most common genre tags inside a cluster
  paste(names(sort(table(unlist(strsplit(x, ",\\s*"))), decreasing = TRUE))[1:n], collapse = " + ")
profile <- netflix %>% group_by(cluster) %>% group_modify(~ tibble(
  size            = nrow(.x),
  pct_movie       = round(100 * mean(.x$is_movie), 1),
  avg_release_yr  = round(mean(.x$release_year), 1),
  median_duration = median(.x$duration_num),
  avg_maturity    = round(mean(.x$maturity), 2),
  avg_cast_size   = round(mean(.x$n_cast), 1),
  avg_genres      = round(mean(.x$n_genres), 2),
  top_country     = mode_of(.x$main_country[.x$main_country != "Unknown"]),
  top_genres      = top_tags(.x$listed_in),
  top_rating      = mode_of(.x$rating)
)) %>% ungroup() %>% as.data.frame()

profile$label <- with(profile, paste0("C", cluster, ": ",
  ifelse(pct_movie >= 50, "Movies", "TV Shows"), " | ", top_genres, " | ",
  as.character(cut(avg_maturity, c(0, 1.75, 2.5, 3.5, 5), labels = c("Kids", "Older Kids", "Teens", "Adults")))))
cat("\n---- CLUSTER PROFILE ----\n"); print(profile)
write.csv(profile, "results/cluster_profile.csv", row.names = FALSE)
netflix$cluster_label <- profile$label[match(netflix$cluster, profile$cluster)]

# Cluster profile heat-map (mean of scaled features per cluster)
Xs <- as.data.frame(X); Xs$cluster <- netflix$cluster
heat <- Xs %>% group_by(cluster) %>% summarise(across(everything(), mean)) %>%
  pivot_longer(-cluster, names_to = "feature", values_to = "mean_z") %>%
  mutate(feature = ifelse(grepl("^g_", feature), paste0("genre: ", unname(genre_lookup[feature])), feature))
p <- ggplot(heat, aes(factor(cluster), feature, fill = mean_z)) + geom_tile(colour = "white") +
  geom_text(aes(label = round(mean_z, 1)), size = 3) +
  scale_fill_gradient2(low = "#221F1F", mid = "white", high = "#E50914", midpoint = 0) +
  labs(title = "Cluster Profile Heat-map (mean z-score)", x = "Cluster", y = NULL, fill = "z")
save_plot(p, "15_cluster_heatmap.png", 9, 8)

# Cluster size and type composition
p <- netflix %>% count(cluster, type) %>%
  ggplot(aes(factor(cluster), n, fill = type)) + geom_col() +
  scale_fill_manual(values = type_cols) +
  labs(title = "Cluster Sizes and Type Composition", x = "Cluster", y = "Titles", fill = NULL) +
  theme(legend.position = "top")
save_plot(p, "16_cluster_sizes.png", 8, 5)

# PCA scatter coloured by cluster
pal <- c("#E50914", "#221F1F", "#4F8EF7", "#F2B134", "#2CA58D", "#9B59B6", "#F5A3A8", "#808080")
p <- ggplot(netflix, aes(PC1, PC2, colour = factor(cluster))) + geom_point(alpha = 0.5, size = 1.1) +
  scale_colour_manual(values = pal[1:k_final]) +
  labs(title = paste0("K-Means Clusters (k = ", k_final, ") on PCA Plane"),
       x = paste0("PC1 (", round(var_exp[1], 1), "%)"),
       y = paste0("PC2 (", round(var_exp[2], 1), "%)"), colour = "Cluster") +
  guides(colour = guide_legend(override.aes = list(alpha = 1, size = 3)))
save_plot(p, "17_pca_clusters.png", 9, 6)

# Cluster by top countries
top5c <- netflix %>% filter(main_country != "Unknown") %>% count(main_country, sort = TRUE) %>%
  slice_head(n = 6) %>% pull(main_country)
p <- netflix %>% filter(main_country %in% top5c) %>% count(main_country, cluster) %>%
  ggplot(aes(main_country, n, fill = factor(cluster))) + geom_col(position = "fill") +
  scale_fill_manual(values = pal[1:k_final]) + scale_y_continuous(labels = scales::percent) +
  labs(title = "Cluster Mix in the Top-6 Countries", x = NULL, y = "Share", fill = "Cluster")
save_plot(p, "18_cluster_by_country.png", 9, 5.5)

# ------------------------------------------------------------------------------
# STAGE 9 : BONUS - PREDICTING "MOVIE vs TV SHOW" (model comparison)
# ------------------------------------------------------------------------------
# Duration is removed (it would give the answer away). Genre tags containing
# "TV" / "Movies" are also removed to avoid target leakage.
leak     <- grep("TV|Movies", grep("^g_", names(netflix), value = TRUE), value = TRUE)
pred_cols <- setdiff(c("release_year", "year_added", "content_age", "n_genres", "n_cast",
                       "has_director", "maturity", grep("^g_", names(netflix), value = TRUE)), leak)
cls <- netflix[, pred_cols]; cls$type <- factor(netflix$type, levels = c("TV Show", "Movie"))

idx_tr <- createDataPartition(cls$type, p = 0.8, list = FALSE)
train  <- cls[idx_tr, ]; test <- cls[-idx_tr, ]
cat("\nTrain rows:", nrow(train), "| Test rows:", nrow(test), "\n")

# 9.1 Logistic regression (GLM)
glm_fit  <- glm(type ~ ., data = train, family = binomial)
glm_prob <- predict(glm_fit, newdata = test, type = "response")
glm_pred <- factor(ifelse(glm_prob > 0.5, "Movie", "TV Show"), levels = levels(test$type))
cm_glm   <- confusionMatrix(glm_pred, test$type, positive = "Movie")
cat("\n---- Logistic Regression ----\n"); print(cm_glm)

# 9.2 Random Forest
rf_fit  <- randomForest(type ~ ., data = train, ntree = 300, importance = TRUE)
rf_pred <- predict(rf_fit, newdata = test)
cm_rf   <- confusionMatrix(rf_pred, test$type, positive = "Movie")
cat("\n---- Random Forest ----\n"); print(cm_rf)

imp <- data.frame(feature = rownames(importance(rf_fit)),
                  gini = importance(rf_fit)[, "MeanDecreaseGini"]) %>% arrange(desc(gini)) %>% slice_head(n = 12)
p <- ggplot(imp, aes(reorder(feature, gini), gini)) + geom_col(fill = nf_red) + coord_flip() +
  labs(title = "Random Forest - Variable Importance", x = NULL, y = "Mean decrease in Gini")
save_plot(p, "19_rf_importance.png", 8, 5.5)

model_cmp <- data.frame(
  Model = c("Logistic Regression", "Random Forest"),
  Accuracy    = round(c(cm_glm$overall["Accuracy"], cm_rf$overall["Accuracy"]), 4),
  Sensitivity = round(c(cm_glm$byClass["Sensitivity"], cm_rf$byClass["Sensitivity"]), 4),
  Specificity = round(c(cm_glm$byClass["Specificity"], cm_rf$byClass["Specificity"]), 4),
  F1          = round(c(cm_glm$byClass["F1"], cm_rf$byClass["F1"]), 4),
  Kappa       = round(c(cm_glm$overall["Kappa"], cm_rf$overall["Kappa"]), 4))
cat("\n---- MODEL COMPARISON ----\n"); print(model_cmp)
write.csv(model_cmp, "results/model_comparison.csv", row.names = FALSE)
write.csv(as.data.frame(cm_rf$table),  "results/confusion_rf.csv",  row.names = FALSE)
write.csv(as.data.frame(cm_glm$table), "results/confusion_glm.csv", row.names = FALSE)

p <- model_cmp %>% pivot_longer(c(Accuracy, F1, Kappa), names_to = "metric", values_to = "value") %>%
  ggplot(aes(Model, value, fill = metric)) + geom_col(position = "dodge") +
  geom_text(aes(label = round(value, 3)), position = position_dodge(0.9), vjust = -0.4, size = 3.5) +
  scale_fill_manual(values = c(nf_red, nf_dark, nf_grey)) + ylim(0, 1.05) +
  labs(title = "Model Comparison: Predicting Movie vs TV Show", x = NULL, y = "Score", fill = NULL) +
  theme(legend.position = "top")
save_plot(p, "20_model_comparison.png", 8, 5)

# ------------------------------------------------------------------------------
# STAGE 10 : SAVE RESULTS (used by the Shiny app and the report)
# ------------------------------------------------------------------------------
write.csv(netflix, "data/netflix_clean.csv", row.names = FALSE, fileEncoding = "UTF-8")
key <- data.frame(name = c("rows_raw", "rows_clean", "n_features", "k_final", "silhouette_best",
                           "between_ss_pct", "sil_kmeans_sample", "sil_hier_sample",
                           "var_pc1", "var_pc2", "pct_movie", "year_min", "year_max"),
                  value = c(n_raw, nrow(netflix), length(feature_cols), k_final, max(sil),
                            100 * km$betweenss / km$totss, sil_km, sil_hc,
                            var_exp[1], var_exp[2], 100 * mean(netflix$is_movie),
                            min(netflix$release_year), max(netflix$release_year)))
write.csv(key, "results/key_numbers.csv", row.names = FALSE)
write.csv(data.frame(k = ks, wss = wss, silhouette = sil), "results/k_selection.csv", row.names = FALSE)

cat("\nDONE. Charts -> /charts | Tables -> /results | Clean data -> /data\n")
sink()
