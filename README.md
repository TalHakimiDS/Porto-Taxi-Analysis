# Porto Taxi: Popular Sub-Route Mining with Spark

Finds the **most popular long sub-routes**, hotspots and anomalous routes in the [Porto taxi trajectory dataset](https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i) (1.71M trips, GPS point every 15 s). It uses distributed PySpark and **approximate data structures**, and is designed to run on Google Cloud Dataproc on a $50 budget.

Course project, *Big Data & Cloud Computing* (B.Sc. Data Science).

## Approach

1. **Preprocessing** (`01_prep_clean.ipynb`, `Stage_01/`): parse `POLYLINE`, remove GPS noise, missing data and outliers (winsorizing), and run statistical EDA.
2. **Spatial encoding** (`02_spatial_encoding.ipynb`): map every GPS point to an **H3 hexagon (resolution 9)** and collapse repeats. Each trip becomes a sequence of cells.
3. **Sub-route mining**, three independent methods:
   | Method | Notebook | Approximate structure | Distributed |
   |---|---|---|---|
   | Graph / n-gram counting | `03a_graph_subroutes.ipynb` | **Count-Min Sketch** (d=5, w=65,536), custom `mapPartitions` + matrix reduce (no PySpark CMS API) | yes |
   | Suffix array + LCP | `03b_suffix_subroutes.ipynb`, `sa_lib.py` | **HyperLogLog** (p=12) for distinct-trip support | no |
   | Clustering | `03c_clustering_subroutes.ipynb` | **MinHash-LSH** (Jaccard, 4 tables) + neighbourhood consensus | yes |
4. **Comparison** (`05_comparison.ipynb`): runtime, memory and agreement between the methods.
5. **Maps** (`04_demo_map.ipynb`): Folium layers for the top routes, a hotspot heatmap and a method-comparison overlay (`out/map_*.html`).

## Results (local 20K-trip sample, min-support 88 trips)

| Method | Runtime | Structure memory | Maximal sub-routes |
|---|---|---|---|
| Graph + Count-Min Sketch | 53.5 s | 2.6 MB | 577 |
| Suffix array + HyperLogLog | 7.2 s | 10.0 MB | 577 |
| MinHash-LSH clustering | 60.8 s | 0.6 MB | 167 |

- The graph and suffix-array methods returned **identical result sets** (Jaccard = 1.0). They are independent algorithms, so each one validates the other.
- LSH clustering is a heuristic that finds *similar* rather than identical routes (precision 0.53 and recall 0.15 against the exact methods). It complements them rather than replacing them.

## Running

The notebooks run locally on `sample_train.csv` (`USE_SAMPLE = True`). For the full dataset, set `USE_SAMPLE = False` and point the paths to your `gs://` bucket. On Dataproc we used an ephemeral cluster with at least 5 nodes and spot workers.

`train.csv` is stored with Git LFS: run `git lfs pull` after cloning.

## Tech

PySpark · H3 · NumPy · pandas · Folium · Google Cloud Dataproc / GCS
