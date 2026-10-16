# FriendFeed User Archetypes and Service Co-adoption

Data Mining I (1DL360) project: Elias Kassie, autumn 2026.

This project uses the FriendFeed dataset (activity from 1 August to 30 September 2010) to answer two questions:

1. **User archetypes (clustering):** can we find distinct types of FriendFeed users from how they publish, interact and connect?
2. **Service co-adoption (association rules):** which external platforms (Twitter, blogs, Flickr, Digg …) do users link together?

---

## Main results

### Question 1: six user archetypes (k-means, k = 6)

| Archetype | Share of users | What characterises it |
|---|---|---|
| Low-activity automated posters | 48.9% | few posts, all from external tools, no interaction |
| Cross-posting broadcasters | 30.0% | many posts from several sources, the most linked services and followers, but no interaction |
| Image sharers | 7.9% | 92% of posts contain an image |
| Native casual users | 6.5% | every post typed directly on FriendFeed, but very few posts |
| Video sharers | 3.5% | every post contains a video |
| Conversationalists | 3.2% | almost all of the commenting and liking; also the most followed |

Three independent criteria point to k = 6:
- the highest silhouette (0.272);
- the most stable result across random starts (ARI ≥ 0.99);
- the best agreement with Ward hierarchical clustering (ARI 0.54).

The silhouette values are modest, so the boundaries between archetypes are soft.

### Question 2: services are linked in bundles (Apriori / FP-Growth, minsup = 5%)

- The strongest bundle is **social bookmarking: Digg, Delicious and StumbleUpon**. For example, {digg, delicious} → {stumbleupon} has lift 3.26 and is backed by 13,810 users.
- A broader **content-sharing bundle** is blog, Flickr, YouTube and Delicious together with Twitter.
- **The Twitter trap:** 85% of users link Twitter, so rules that predict Twitter reach confidence 0.93 but lift only about 1.09. For that reason, rules are ranked by **lift**, not by confidence.
- The rules are stable: 95.9% of them reappear when the data is split into two random halves, and their lift values correlate at 0.999.

---

## Repository structure

```
Project/
├── Code/
│   ├── 01_load_raw.ipynb            # read and clean the raw CSVs, save as Parquet
│   ├── 02_build_features.ipynb      # build one row of features per user
│   ├── 03_eda_first_models.ipynb    # exploratory analysis and first models (preliminary report)
│   ├── 04_clustering.ipynb          # final clustering: user archetypes (Question 1)
│   └── 05_association_rules.ipynb   # association rules on linked services (Question 2)
├── figures/                         # all plots used in the reports
├── Project_Proposal__Data_Mining_I.pdf
├── Preliminary_Report_FriendFeed.pptx
├── Final_Presentation_FriendFeed.pptx
└── DM1-Project-Info-2026.pdf        # course project instructions
```

**The data is not included.** The raw CSVs (about 7.5 GB) and the cleaned Parquet files are far above GitHub's 100 MB file limit, so `.gitignore` excludes them. See [Reproducing the results](#reproducing-the-results) below.

---

## Process

The notebooks follow the KDD process: selection → preprocessing → transformation → data mining → interpretation. Each notebook ends its sections with **Findings** cells that explain the decisions made.

### 01: Loading and cleaning the raw data
- The raw files have no header row, use three different separators (`|`, `,`, tab) and three different ways of marking a missing value (`null`, `\N`, empty). Each file was therefore read with its own settings, using DuckDB.
- **The documentation does not match the data, and this was checked against the files.** The comments file has 9 columns, not the 13 the README lists. The follower file has its columns in the opposite order to the README; this was confirmed by checking that famous accounts appear in the "followed" column.
- Cleaning steps:
  - removed duplicate rows (users, services, likes, follow edges);
  - skipped 72 malformed lines (0.005%);
  - dropped empty columns (GeoX/GeoY);
  - flagged posts dated outside the study window, including 1970 timestamps.
- Groups (6.9% of accounts) were kept in the cleaned file but excluded from the analysis, because the research questions are about individual people.
- Output: compact Parquet files (for example, `users.csv` at 43 MB becomes 15.6 MB).

### 02: Building the user feature table
- All activity was restricted to the study window (1 Aug – 30 Sep 2010).
- Population: **201,335 individual users who posted at least once** (98.8% of active users).
- The 15 features per user cover:
  - posting: number of posts, number of sources, and the share of posts that are native, contain an image, or contain a video;
  - comments: on others, on own posts, received;
  - likes: given and received;
  - network: followers, following;
  - number of linked services;
  - whether the profile has a description.
- Half of all comments are on the user's *own* post. These were separated from comments on others so that "conversation" is measured correctly.

### 03: Exploratory analysis and first models
- All counts are extremely right-skewed (median 10 posts, maximum 166,965), so they were transformed with `log1p` and then standardised with z-scores before Euclidean k-means.
- About 94% of users never comment or like. Outliers were **kept**, because very active or automated accounts are the subject of the study.
- The four interaction features are strongly correlated (Spearman 0.67–0.76). This inflated the silhouette at k = 2, a problem fixed in notebook 04.
- A first Apriori run on the services data revealed the Twitter trap.

### 04: Clustering (Question 1)
- PCA was used as a diagnostic. It confirmed that the four interaction features carry the same information, so they were merged into a single `n_interaction` feature. That left 10 features.
- k from 2 to 8 was evaluated with SSE (no clear elbow) and silhouette (best at k = 6).
- Stability was checked across 5 random seeds using the Adjusted Rand Index.
- The result was checked against a second algorithm: Ward hierarchical clustering on a sample of 5,000 users.
- The clusters were named from their median profiles and visualised with a heatmap and a PCA projection.
- Output: `user_clusters.parquet` (user id, cluster, archetype).

### 05: Association rules (Question 2)
- Each user's basket is the set of services they linked. Users with only one service were removed, because a one-item basket cannot produce a rule. This left 198,824 baskets over 60 services.
- minsup = 5% was chosen with a sensitivity analysis from 2% to 20%. At 5%, every rule is backed by at least 9,941 users.
- Apriori and FP-Growth give identical results: 348 frequent itemsets. These reduce to 348 closed and 139 maximal itemsets.
- Rules were kept if confidence ≥ 0.5 and lift > 1, giving 721 rules ranked by lift. A pairwise lift matrix and a split-half stability check complete the analysis.

---

## Reproducing the results

### 1. Environment
Python 3.11 with:

```
pip install -r requirements.txt

```

The versions used were pandas 2.3, numpy 2.4, scikit-learn 1.9, scipy 1.17, matplotlib 3.10, duckdb 1.5, pyarrow 25 and mlxtend 0.25. The machine had 16 GB of RAM, which is enough because DuckDB reads the CSVs out of core.

### 2. Data
Get the FriendFeed v2 dataset (provided through the course) and put the raw CSV files in a folder named `Data Source_Friendfeed study/` inside `Project/`:

```
Project/
└── Data Source_Friendfeed study/
    ├── users.csv
    ├── services.csv
    ├── entries1.csv, entries2.csv, entries3.csv
    ├── commentAugSept.csv
    ├── likes.csv
    ├── subscriptions.csv
    └── followingAugSept.csv
```

### 3. Run the notebooks in order
Open the notebooks from the `Code/` folder. Paths are set relative to the project (`Path.cwd().parent`), so the notebooks must be started from inside `Code/`. Run them in this order:

1. `01_load_raw.ipynb` → writes the cleaned Parquet files to `Project/data/`
2. `02_build_features.ipynb` → writes `data/user_features.parquet`
3. `03_eda_first_models.ipynb`
4. `04_clustering.ipynb` → writes `data/user_clusters.parquet`
5. `05_association_rules.ipynb`

The figures are saved to `figures/`. All random steps use fixed seeds (`random_state=42`), so the results can be reproduced.

---

## Limitations
- Silhouette values of about 0.2–0.3 show that the archetypes are tendencies along a continuum, not sharply separated groups.
- Follower, following and service counts come from snapshots taken in October 2010, about two weeks after the activity window.
- Services with less than 5% support cannot appear in any rule (the rare-item problem).
- Post and comment text was not analysed, because text mining is outside the course scope.
