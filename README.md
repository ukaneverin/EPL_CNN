# EPL_CNN
This repository provides implementation for the article *Beyond classification: Whole slide tissue histopathology analysis by end-to-end part learning* by **Xie et al. 2021**.

## Guideline
`example_run.py` is an example script for running the data processing, training and evaluation for whole slide image analysis. 

Please use `python example_run.py --help` to see complete set of parameters and their descriptions.

Whole slide images should be tiled into a library of image patches with the following format:

SlideID | Split | target | x | y 
------------ | ------------- | -------------| -------------| ------------- 
1 | train | 1 | 0 | 0 
2 | train | 1 | 224| 0 
3 | train | 0 | 448 | 0
... | ... | ... | ... 
324 | validation | 0 | 512 | 2240
... | ... | ... | ... 
556 | validation | 1 | 5120 | 2240 

Use the following command to run:

`python example_run.py --stage $stage$`

`args.stage` specifies which 

The following will be generated in the output folder:
* convergence.csv
  * a file containing training loss, training concordance index, and validation condorance index over training epochs
* /clustering_grid_top
  * a folder where a clustering visualization for top 20 tiles of each cluster is displayed and saved as a `.png` 

## Python Dependencies
* torch 1.8.1
  * torchvision 0.9.1
* lifelines 0.23.8
* openslide 1.1.1
  * *Note: We recommend modifying openslide to correct for memory leak issue. Please see https://github.com/openslide/openslide-python/issues/24 for more information.*

## License
This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE.md) for details. (c) MSK

## Cite
If you find our work useful, please consider citing our [EPIC-Survival Paper](https://openreview.net/pdf?id=JSSwHS_GU63):
```
@inproceedings{muhammad2021epic,
  title={EPIC-Survival: End-to-end Part Inferred Clustering for Survival Analysis, with Prognostic Stratification Boosting},
  author={Muhammad, Hassan and Xie, Chensu and Sigel, Carlie S and Doukas, Michael and Alpert, Lindsay and Simpson, Amber Lea and Fuchs, Thomas J},
  booktitle={Medical Imaging with Deep Learning},
  year={2021}
}
```
