# Consolidated Kaggle Playground Series

Welcome to my consolidated repository for [Kaggle Playground Series](https://www.kaggle.com/competitions?hostSegmentIdFilter=8) competitions! 

This repository serves as a centralized hub for all my code, models, and experiments across various Playground Series episodes.

## 📁 Repository Structure

The repository is organized into separate directories for each competition episode, following a standardized template to maintain a clean and consistent workflow.

```text
kaggle-pgs/
├── pgs-sXeY/                 # Directory for a specific competition (e.g., Season 6, Episode 8)
│   ├── dataset/              # Raw and processed data (ignored by git)
│   ├── code/                 # Notebooks and scripts
│   │   ├── model-01/         # Specific model/approach
│   │   │   ├── kernel-metadata.json  # Kaggle API metadata for pushing the kernel
│   │   │   └── notebook.ipynb        # The actual Jupyter notebook
│   │   └── model-02/         # Alternative models or ensembles
│   ├── submission/           # Generated submission.csv files (ignored by git)
│   └── README.md             # Competition-specific details, scores, and insights
├── .gitignore                # Global gitignore excluding datasets and submissions
└── README.md                 # This file
```

## 🏆 Competitions Tracked

*This table will be updated as I participate in more competitions.*

| Competition | Folder | Public LB | Private LB | Private Rank | Evaluation Metric |
| :---: | :---: | :---: | :---: | :---: | :---: |
| *Playground S6E8* | [`pgs-s6e8/`](./pgs-s6e8) | 0.96657 | 0.96626 | 1352 | AUC-ROC |
| *Playground S6E6* | [`pgs-s6e6/`](./pgs-s6e6) | 0.95841 | 0.95742 | 1752 | Balanced Accuracy |
| *Playground S6E5* | - | 0.94623 | 0.94655 | 1896 | AUC-ROC |
| *Playground S6E4* | - | 0.96703 | 0.96900 | 1614 | Balanced Accuracy |
| *Playground S5E10* | - | 0.05581 | 0.05609 | 2793 | RMSE |
| *Playground S5E8* | - | 0.89330 | 0.89110 | 2690 | AUC-ROC |
| *Playground S4E11* | - | 0.93987 | 0.93881 | 1577 | Accuracy Score |