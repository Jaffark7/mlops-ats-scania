# Data

The original Scania APS training and test CSV files are intentionally not tracked
in git. Download them from the UCI Machine Learning Repository page for
`APS Failure at Scania Trucks`, then place them here:

```text
data/aps_failure_training_set.csv
data/aps_failure_test_set.csv
```

The files contain 20 metadata lines before the CSV header; the project loaders
handle that with `skiprows=20`.
