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

`args.stage` specifies which stage of the pipeline the scipts should run: `'train'`, `'val'`, `'test'`

Check `makedata.py` for the working directory and the outputs that will be saved.

## Python Dependencies
* torch 1.4.0
  * torchvision 0.5.0
* openslide 1.1.1
  * *Note: We recommend modifying openslide to correct for memory leak issue. Please see https://github.com/openslide/openslide-python/issues/24 for more information.*

## License
This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE.md) for details. (c) MSK

## Cite
If you find our work useful, please consider citing our [EPL Paper](http://proceedings.mlr.press/v121/xie20a/xie20a.pdf):
```

@InProceedings{pmlr-v121-xie20a,
  title = 	 {Beyond Classification: Whole Slide Tissue Histopathology Analysis By End-To-End Part Learning},
  author =       {Xie, Chensu and Muhammad, Hassan and Vanderbilt, Chad M. and Caso, Raul and Yarlagadda, Dig Vijay Kumar and Campanella, Gabriele and Fuchs, Thomas J.},
  booktitle = 	 {Proceedings of the Third Conference on Medical Imaging with Deep Learning},
  pages = 	 {843--856},
  year = 	 {2020},
  editor = 	 {Arbel, Tal and Ben Ayed, Ismail and de Bruijne, Marleen and Descoteaux, Maxime and Lombaert, Herve and Pal, Christopher},
  volume = 	 {121},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {06--08 Jul},
  publisher =    {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v121/xie20a/xie20a.pdf},
  url = 	 {https://proceedings.mlr.press/v121/xie20a.html},
  abstract = 	 {An emerging technology in cancer care and research is the use of histopathology whole slide images (WSI). Leveraging computation methods to aid in WSI assessment poses unique challenges. WSIs, being extremely high resolution giga-pixel images, cannot be directly processed by convolutional neural networks (CNN) due to huge computational cost. For this reason, state-of-the-art methods for WSI analysis adopt a two-stage approach where the training of a tile encoder is decoupled from the tile aggregation. This results in a trade-off between learning diverse and discriminative features. In contrast, we propose end-to-end part learning (EPL) which is able to learn diverse features while ensuring that learned features are discriminative. Each WSI is modeled as consisting of $k$ groups of tiles with similar features, defined as parts. A loss with respect to the slide label is backpropagated through an integrated CNN model to $k$ input tiles that are used to represent each part. Our experiments show that EPL is capable of clinical grade prediction of prostate and basal cell carcinoma. Further, we show that diverse discriminative features produced by EPL succeeds in multi-label classification of lung cancer architectural subtypes. Beyond classification, our method provides rich information of slides for high quality clinical decision support.}
}

```
