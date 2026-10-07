# ==============================================================================
#  NETFLIX CONTENT CLUSTERING - SHINY DASHBOARD
#  Run:  1) first run  netflix_analysis.R  once (creates data/netflix_clean.csv)
#        2) open this file in RStudio and click  "Run App"
# ==============================================================================
library(shiny)
library(dplyr)
library(ggplot2)

nf_red <- "#E50914"; nf_dark <- "#221F1F"
pal <- c("#E50914", "#FFFFFF", "#4F8EF7", "#F2B134", "#2CA58D", "#9B59B6",
         "#F5A3A8", "#808080", "#8B4513", "#17BECF")
type_cols <- c("Movie" = nf_red, "TV Show" = nf_dark)

# ---- data ---------------------------------------------------------------------
df <- read.csv("data/netflix_clean.csv", stringsAsFactors = FALSE, fileEncoding = "UTF-8")
df$maturity_label <- factor(df$maturity_label, levels = c("Kids", "Older Kids", "Teens", "Adults"))
feature_cols <- c("is_movie", "duration_z", "release_year", "content_age_log", "n_genres",
                  "n_cast", "has_director", "maturity", "year_added",
                  grep("^g_", names(df), value = TRUE))
X       <- scale(df[, feature_cols])
pca     <- prcomp(X)
df$PC1  <- pca$x[, 1]; df$PC2 <- pca$x[, 2]
var_exp <- 100 * pca$sdev^2 / sum(pca$sdev^2)
k_tab   <- if (file.exists("results/k_selection.csv")) read.csv("results/k_selection.csv") else NULL
if (dir.exists("charts")) addResourcePath("charts", "charts")

top_tags <- function(x, n = 2)
  paste(names(sort(table(unlist(strsplit(x, ",\\s*"))), decreasing = TRUE))[1:n], collapse = " + ")

theme_nf <- function(base = 13) theme_minimal(base_size = base) + theme(
  plot.background  = element_rect(fill = "#141414", colour = NA),
  panel.background = element_rect(fill = "#141414", colour = NA),
  panel.grid.major = element_line(colour = "#2f2f2f"), panel.grid.minor = element_blank(),
  text = element_text(colour = "#f5f5f1"), axis.text = element_text(colour = "#b3b3b3"),
  plot.title = element_text(face = "bold"))

card <- function(value, label)
  div(class = "stat-card", div(class = "stat-value", value), div(class = "stat-label", label))

# ---- UI -----------------------------------------------------------------------
ui <- fluidPage(
  tags$head(tags$style(HTML("
    body { background:#141414; color:#f5f5f1; font-family: 'Helvetica Neue', Arial, sans-serif; }
    h1,h2,h3,h4 { color:#f5f5f1; }
    .app-title { color:#E50914; font-weight:800; letter-spacing:2px; margin:18px 0 4px 0; }
    .app-sub { color:#b3b3b3; margin-bottom:18px; }
    .nav-tabs > li > a { color:#b3b3b3; background:#222; border:none; margin-right:4px; }
    .nav-tabs > li.active > a, .nav-tabs > li.active > a:focus, .nav-tabs > li.active > a:hover
      { background:#E50914; color:#fff; border:none; }
    .nav-tabs { border-bottom:2px solid #E50914; }
    .well { background:#222; border:none; color:#f5f5f1; }
    .stat-card { background:#222; border-left:5px solid #E50914; padding:14px 18px; margin-bottom:14px; border-radius:4px; }
    .stat-value { font-size:30px; font-weight:800; }
    .stat-label { color:#b3b3b3; font-size:13px; text-transform:uppercase; letter-spacing:1px; }
    table { color:#f5f5f1; } .table-striped > tbody > tr:nth-of-type(odd) { background:#1c1c1c; }
    .form-control, .selectize-input { background:#333 !important; color:#fff !important; border:none !important; }
    .selectize-dropdown { background:#333; color:#fff; }
    .note { color:#b3b3b3; font-size:13px; }
    img.heat { max-width:100%; border-radius:4px; background:#fff; }
  "))),
  h1(class = "app-title", "NETFLIX CONTENT CLUSTERING"),
  div(class = "app-sub", "R Mini Project | K-Means, Hierarchical Clustering & PCA on the Netflix catalogue"),
  tabsetPanel(
    # ---- Tab 1 --------------------------------------------------------------------
    tabPanel("Overview",
      br(),
      fluidRow(
        column(3, card(format(nrow(df), big.mark = ","), "Titles analysed")),
        column(3, card(sum(df$type == "Movie"), "Movies")),
        column(3, card(sum(df$type == "TV Show"), "TV Shows")),
        column(3, card(paste(min(df$release_year), "-", max(df$release_year)), "Release years"))),
      fluidRow(
        column(5, plotOutput("ov_type", height = 320)),
        column(7, plotOutput("ov_added", height = 320))),
      h4("Sample of the cleaned dataset"),
      tableOutput("ov_table")),
    # ---- Tab 2 --------------------------------------------------------------------
    tabPanel("EDA Explorer",
      br(),
      sidebarLayout(
        sidebarPanel(width = 3,
          checkboxGroupInput("type", "Type", c("Movie", "TV Show"), c("Movie", "TV Show")),
          sliderInput("yr", "Release year", min(df$release_year), max(df$release_year),
                      c(2000, max(df$release_year)), sep = ""),
          selectInput("mat", "Audience maturity", c("All", levels(df$maturity_label))),
          p(class = "note", "Charts update instantly when you change a filter.")),
        mainPanel(width = 9,
          fluidRow(column(6, plotOutput("eda_year", height = 300)),
                   column(6, plotOutput("eda_country", height = 300))),
          fluidRow(column(6, plotOutput("eda_genre", height = 300)),
                   column(6, plotOutput("eda_rating", height = 300)))))),
    # ---- Tab 3 --------------------------------------------------------------------
    tabPanel("Live Clustering",
      br(),
      sidebarLayout(
        sidebarPanel(width = 3,
          sliderInput("k", "Number of clusters (k)", 2, 10, 6, step = 1),
          uiOutput("km_stats"),
          p(class = "note", "Move the slider and K-Means re-runs live. The project's chosen value is k = 6 (saved best-of-50 solution).")),
        mainPanel(width = 9,
          fluidRow(column(8, plotOutput("pca_plot", height = 380)),
                   column(4, plotOutput("size_plot", height = 380))),
          fluidRow(column(6, plotOutput("elbow_plot", height = 250)),
                   column(6, h4("Cluster summary"), tableOutput("cl_table")))))),
    # ---- Tab 4 --------------------------------------------------------------------
    tabPanel("Cluster Profiles",
      br(),
      fluidRow(
        column(5,
          selectInput("prof_cl", "Choose a cluster (k = 6 final model)",
                      sort(unique(df$cluster_label)), width = "100%"),
          uiOutput("prof_cards"),
          h4("Random titles from this cluster"), tableOutput("prof_titles")),
        column(7, h4("Cluster heat-map (mean z-score of each feature)"),
               if (file.exists("charts/15_cluster_heatmap.png"))
                 tags$img(class = "heat", src = "charts/15_cluster_heatmap.png"))),
      hr(),
      h4("Which cluster is my favourite title in?"),
      textInput("search", NULL, placeholder = "Type a title, e.g. Stranger, Narcos, Naruto ..."),
      tableOutput("search_res")),
    # ---- Tab 5 --------------------------------------------------------------------
    tabPanel("About",
      br(),
      h4("About this project"),
      p("Dataset: Kaggle - Netflix Movies and TV Shows (", nrow(df), " titles after cleaning)."),
      p("Pipeline: data cleaning -> feature engineering -> EDA -> scaling -> K-Means -> Hierarchical (Ward) -> PCA -> cluster profiling."),
      p("Features used: movie/TV flag, standardised duration, release year, content age, genres count,",
        " cast size, director present, maturity level, year added and the 10 most common genre tags."))
  )
)

# ---- SERVER -------------------------------------------------------------------
server <- function(input, output, session) {

  # Overview
  output$ov_type <- renderPlot({
    d <- df %>% count(type)
    ggplot(d, aes(type, n, fill = type)) + geom_col(width = 0.6) +
      geom_text(aes(label = n), vjust = -0.5, colour = "white", fontface = "bold") +
      scale_fill_manual(values = c("Movie" = nf_red, "TV Show" = "#808080")) +
      labs(title = "Movies vs TV Shows", x = NULL, y = NULL) +
      theme_nf(14) + theme(legend.position = "none")
  }, bg = "#141414")
  output$ov_added <- renderPlot({
    df %>% count(year_added, type) %>%
      ggplot(aes(year_added, n, fill = type)) + geom_col() +
      scale_fill_manual(values = c("Movie" = nf_red, "TV Show" = "#808080")) +
      labs(title = "Titles added to Netflix per year", x = NULL, y = NULL, fill = NULL) +
      theme_nf(14) + theme(legend.position = "top")
  }, bg = "#141414")
  output$ov_table <- renderTable(
    head(df[, c("title", "type", "main_country", "release_year", "rating", "duration", "main_genre", "cluster")], 8))

  # EDA explorer
  filtered <- reactive({
    d <- df %>% filter(type %in% input$type, release_year >= input$yr[1], release_year <= input$yr[2])
    if (input$mat != "All") d <- d %>% filter(maturity_label == input$mat)
    validate(need(nrow(d) > 0, "No titles match the selected filters."))
    d
  })
  dark <- theme_nf(13)
  output$eda_year <- renderPlot({
    filtered() %>% count(release_year, type) %>%
      ggplot(aes(release_year, n, colour = type)) + geom_line(linewidth = 1.1) +
      scale_colour_manual(values = c("Movie" = nf_red, "TV Show" = "#cccccc")) +
      labs(title = "Titles by release year", x = NULL, y = NULL, colour = NULL) + dark +
      theme(legend.position = "top")
  }, bg = "#141414")
  output$eda_country <- renderPlot({
    filtered() %>% filter(main_country != "Unknown") %>% count(main_country, sort = TRUE) %>%
      slice_head(n = 10) %>%
      ggplot(aes(reorder(main_country, n), n)) + geom_col(fill = nf_red) + coord_flip() +
      labs(title = "Top countries", x = NULL, y = NULL) + dark
  }, bg = "#141414")
  output$eda_genre <- renderPlot({
    g <- as.data.frame(table(unlist(strsplit(filtered()$listed_in, ",\\s*"))))
    g %>% arrange(desc(Freq)) %>% slice_head(n = 10) %>%
      ggplot(aes(reorder(Var1, Freq), Freq)) + geom_col(fill = "#b3b3b3") + coord_flip() +
      labs(title = "Top genres", x = NULL, y = NULL) + dark
  }, bg = "#141414")
  output$eda_rating <- renderPlot({
    filtered() %>% count(rating, type) %>%
      ggplot(aes(reorder(rating, -n), n, fill = type)) + geom_col() +
      scale_fill_manual(values = c("Movie" = nf_red, "TV Show" = "#808080")) +
      labs(title = "Rating distribution", x = NULL, y = NULL, fill = NULL) + dark +
      theme(legend.position = "top", axis.text.x = element_text(angle = 45, hjust = 1))
  }, bg = "#141414")

  # Live clustering
  # k = 6 re-uses the best solution saved by netflix_analysis.R (nstart = 50);
  # any other k is computed live with K-Means.
  k_saved <- max(df$cluster)
  totss   <- sum(scale(X, scale = FALSE)^2)
  assign_cl <- reactive({
    if (input$k == k_saved) df$cluster
    else { set.seed(123); kmeans(X, centers = input$k, nstart = 25, iter.max = 50)$cluster }
  })
  clustered <- reactive({ d <- df; d$cl <- factor(assign_cl()); d })
  output$km_stats <- renderUI({
    cl  <- assign_cl()
    wss <- sum(sapply(unique(cl), function(c) sum(scale(X[cl == c, , drop = FALSE], scale = FALSE)^2)))
    tagList(card(paste0(round(100 * (totss - wss) / totss, 1), "%"), "Between-SS / Total-SS"),
            card(format(round(wss), big.mark = ","), "Total within-cluster SS"))
  })
  output$pca_plot <- renderPlot({
    ggplot(clustered(), aes(PC1, PC2, colour = cl)) + geom_point(alpha = 0.55, size = 1.2) +
      scale_colour_manual(values = pal) +
      labs(title = paste0("K-Means (k = ", input$k, ") on the PCA plane"),
           x = paste0("PC1 (", round(var_exp[1], 1), "%)"),
           y = paste0("PC2 (", round(var_exp[2], 1), "%)"), colour = "Cluster") +
      theme_nf(13) + guides(colour = guide_legend(override.aes = list(alpha = 1, size = 3)))
  }, bg = "#141414")
  output$size_plot <- renderPlot({
    clustered() %>% count(cl, type) %>%
      ggplot(aes(cl, n, fill = type)) + geom_col() +
      scale_fill_manual(values = c("Movie" = nf_red, "TV Show" = "#808080")) +
      labs(title = "Cluster sizes", x = "Cluster", y = NULL, fill = NULL) +
      theme_nf(13) + theme(legend.position = "top")
  }, bg = "#141414")
  output$elbow_plot <- renderPlot({
    req(k_tab)
    ggplot(k_tab, aes(k, wss)) + geom_line(colour = nf_red, linewidth = 1.1) + geom_point(colour = "white", size = 2.5) +
      geom_point(data = k_tab[k_tab$k == input$k, ], colour = "#F2B134", size = 5) +
      scale_x_continuous(breaks = k_tab$k) +
      labs(title = "Elbow curve (selected k highlighted)", x = "k", y = "Within-cluster SS") +
      theme_nf(12)
  }, bg = "#141414")
  output$cl_table <- renderTable({
    clustered() %>% group_by(Cluster = cl) %>%
      summarise(Titles = n(), `% Movies` = as.integer(round(100 * mean(is_movie))),
                `Avg year` = as.integer(round(mean(release_year))), `Top genres` = top_tags(listed_in), .groups = "drop") %>%
      as.data.frame()
  })

  # Profiles
  prof <- reactive(df %>% filter(cluster_label == input$prof_cl))
  output$prof_cards <- renderUI({
    d <- prof()
    fluidRow(column(4, card(nrow(d), "Titles")),
             column(4, card(paste0(round(100 * mean(d$is_movie)), "%"), "Movies")),
             column(4, card(round(mean(d$release_year)), "Avg release yr")))
  })
  output$prof_titles <- renderTable({
    set.seed(1); d <- prof(); d <- d[sample(nrow(d), min(10, nrow(d))), ]
    d[, c("title", "type", "main_country", "release_year", "rating")]
  })
  output$search_res <- renderTable({
    req(nchar(trimws(input$search)) >= 3)
    r <- df[grepl(trimws(input$search), df$title, ignore.case = TRUE), c("title", "type", "release_year", "rating", "cluster_label")]
    validate(need(nrow(r) > 0, "No title found."))
    head(r, 15)
  })
}

shinyApp(ui, server)
