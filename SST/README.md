# Running instructions on SST-2/any generic dataset with binary labels.
# Just remember the format is sentence \t label


Once you build the conda environment
```
from data_utils import IMDBDataset
import pickle
dataset = IMDBDataset(path_to_your_data)
pickle.dump(dataset, open("dataset.pkl","wb"))
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
a = pickle.load(open("dataset.pkl","rb"))
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
dataset = pickle.load(open("dataset","rb"))
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
