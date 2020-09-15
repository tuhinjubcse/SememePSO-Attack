# Running instructions on SST-2 dataset
## Data
- Download SST-2 dataset: [sst-2.zip (Tsinghua)](https://cloud.tsinghua.edu.cn/d/b6b35b7b7fdb43c1bf8c/files/?p=%2Fsst-2.zip) or [sst-2.zip (Google Drive)](https://drive.google.com/file/d/1f8Wmj3jqTzdstGdj8x1YDdh4d6axDDrE/view?usp=sharing)
- Download processed SST-2 data for training models: [SST2data.zip (Tsinghua)](https://cloud.tsinghua.edu.cn/d/b6b35b7b7fdb43c1bf8c/files/?p=%2FSST2data.zip) or [SST2data.zip (Google Drive)](https://drive.google.com/file/d/1qV8jnDeFoZgSZlT3pO3jFMoaTIPGIb6G/view?usp=sharing)
## Process Data and Train Model
(You can skip these part if you have downloaded our processed model and data files.)

- Process SST-2 Data
```bash
python data_utils.py
```
- Generate Candidate Substitution Words 
```bash
python gen_pos.py
python lemma.py
python gen_candidates.py
```
- Train BiLSTM Model  
```bash
python train_model.py
```
- Train BERT Model 
```bash 
python SST_BERT.py
```
## Craft Adversarial Examples
- Crafting Adversarial Examples for Bi-LSTM
```bash
python AD_dpso_sem.py
```
- Crafting Adversarial Examples for BERT
```bash
python AD_dpso_sem_bert.py
```
The generated `AD_dpso_sem_bert.pkl` will contain the adversarial examples and original examples as well as their labels.

- To recover adversarial examples in string forms, use the following codes:
```python
from data_utils import IMDBDataset
import pickle
a = pickle.load(open("aux_files/dataset_13837.pkl","rb"))
adv_list = pickle.load(open("AD_dpso_sem_bert.pkl", "rb"))[3]
adversarial_examples = []
for example_id_list in adv_list:
   example_word_list = [a.inv_dict[int(x)] for x in example_id_list if int(x)!=0]
   adversarial_examples.append(example_word_list)
```
- To recover corresponding original examples in string forms, use the following codes:
```python
from data_utils import IMDBDataset
import pickle
dataset = pickle.load(open("aux_files/dataset_xxxx.pkl","rb"))
adv_orig = pickle.load(open("AD_dpso_sem_bert.pkl", "rb"))[2]
test_x = dataset.test_seqs2
original_examples = []
for _id in adv_orig:
   test_example = test_x[_id]
   example_word_list = [dataset.inv_dict[int(x)] for x in test_example if int(x)!=0]
   original_examples.append(example_word_list)
```

## Instructions for running our codes on other similar datasets
1. Firstly, transform your data as {train,valid,test}.tsv as in SST (our github repo provides examples of these TSV files)
2. Then, open up a python terminal, run codes as in "Process Data and Train Model" section.
3. For `SST2data`, you can change your data to our json formats and replace original files under this directory with yours.
All files we provided under the directory are json-like and self-explained. `*ids.py` file are the ids used in train/valid/dev parition, `xxx_inputs.json` includes all the inputs (id start from 0).
 You can change the parameter setting in `SSTconfig.py` if you need to.
4. train the BERT model and generate examples via running codes in "Craft Adversarial Examples"
5. You should find your examples in `AD_dpso_sem_bert.pkl`.
