# Problem definition

ZOC studies zero-shot, open-vocabulary, query-conditioned counting. The system receives an image and a text query and must return the number of physical instances satisfying that query.

The target is not merely a plausible image-level number. The evaluation must test which instances are selected, whether instances are duplicated or merged, and whether a verified-absent query produces zero.
