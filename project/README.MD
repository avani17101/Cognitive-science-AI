## Team JAM

Implementation of paper [From voxels to pixels and back: Self-supervision in natural-image reconstruction from fMRI.Beliy, R., Gaziv, G., Hoogi, A., Strappini, F., Golan, T., & Irani, M. (2019). Advances in Neural Information Processing Systems, 32.](https://proceedings.neurips.cc/paper/2019/hash/7d2be41b1bde6ff8fe45150c37488ebb-Abstract.html) NEURIPS, 2019

## Repository structure
```
├── docs
├── src
	├── models #trained models saved here
	├── main.py #run for train/test
	├── other python files
    ├── data #data files
    ├── KamitaniData #image id mapping csv files
    ├── cfolder where results are saved
├── environment.yml 
├── requirements.txt

```
## Instructions to run the code

* clone repository
* install requirements
```bash
conda env create -f environment.yml
```
* download data from http://brainliner.jp/data/brainliner/Generic_Object_Decoding and http://image-net.org/download
* cd src
* prepare data: run KamitaniData/kamitani_image_prepare.py
```bash
python KamitaniData/kamitani_image_prepare.py
```
* change paths in config/config
* run main.py
```bash
python main.py
```
* result is generated in results folder
* PS: we have provided log files: our training logs in encoderlogs.txt  and encdeclogs.txt.
* results_600ssl are the results obtained when we trained only for unpaired 600 images instead of 50k upaired images.
* results are  the results obtained when we trained for all datasets mentioned.
* Our trained model weights files are available at : https://drive.google.com/drive/folders/1U0yvaQfz1kx-fzrd7iPZTf4h8oXo-JQM?usp=sharing