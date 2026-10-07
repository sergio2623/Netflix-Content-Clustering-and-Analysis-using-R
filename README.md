# Netflix Content Clustering and Analysis using R

R Programming Mini Project: unsupervised learning (K-Means, Hierarchical clustering, PCA) on the Kaggle
**Netflix Movies and TV Shows** dataset, with a supervised bonus (Logistic Regression vs Random Forest)
and an interactive **Shiny dashboard**.

## Folder structure
| File / folder | Purpose |
|---|---|
| `netflix_titles.csv` | Original Kaggle dataset (7,787 titles, 12 columns) |
| `netflix_analysis.R` | Full analysis script (cleaning, EDA, clustering, PCA, profiling, models) |
| `app.R` | Shiny dashboard |
| `install_packages.R` | Installs the required packages |
| `charts/` | All charts produced by the script |
| `results/` | CSV tables and the console output |
| `data/netflix_clean.csv` | Cleaned data with cluster labels (used by the Shiny app) |
| `screenshots/` | Screenshots of the Shiny dashboard |
| `*.docx`, `*.pptx` | Project report and presentation |

## How to run
1. Open RStudio, then `Session > Set Working Directory > To Source File Location` (or open the folder as a project).
2. Run `source("install_packages.R")` once.
3. Run `source("netflix_analysis.R")` (about 1-2 minutes). Charts, tables and cleaned data are regenerated.
4. Open `app.R` and click **Run App** (or `shiny::runApp("app.R")`).

## Key results
* 7,682 titles after cleaning, 19 clustering features.
* K-Means with k = 6 gives six segments: Kids & Family Movies, Adult TV Dramas, Documentaries & Stand-up,
  International TV Shows, Mainstream International Movies & Dramas, Independent Dramas.
* Hierarchical clustering (Ward) confirms the structure; PCA: 33% variance in two components.
* Movie vs TV Show prediction: Random Forest 97.4% accuracy, Logistic Regression 97.0%.

## Author
[Your Name], [Register Number], [Department, College Name]
