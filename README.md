# MultiStratifiedKFold
Allows stratifying by along multiple variables instead of just one, like the original StratifiedKFold. Useful for example in machine learning tasks with a protected attribute (e.g. stratify by dependent variable and race, not just dependent variable).

Usage:
```python
from MultiStratifiedKFold import MutliStratifiedKFold

skf = MultiStratifiedKFold(n_splits = 5)
for train_idx, test_idx in skf.split(X, (race, y)):
  # the rest as in usual sklearn.model_selection.StratifiedKFold
```
