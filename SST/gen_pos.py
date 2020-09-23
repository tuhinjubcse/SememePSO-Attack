import pickle
import nltk


with open('dataset.pkl','rb') as fp:
    dataset=pickle.load(fp)


#from nltk.stem import WordNetLemmatizer
#wnl = WordNetLemmatizer()
#pos_tag=pos_tagger.tag(['what', "'s", 'invigorating', 'about', 'it', 'is', 'that', 'it', 'does', "n't", 'give', 'a', 'damn'])
#print(pos_tag)

train_text=[[dataset.inv_full_dict[t] for t in tt] for tt in dataset.train_seqs]
test_text=[[dataset.inv_full_dict[t] for t in tt] for tt in dataset.test_seqs]
all_pos_tags=[]
test_pos_tags=[]
for text in train_text:
    pos_tags = nltk.pos_tag(text)
    all_pos_tags.append(pos_tags)
for text in test_text:
    pos_tags = nltk.pos_tag(text)
    test_pos_tags.append(pos_tags)
f=open('pos_tags.pkl','wb')
pickle.dump(all_pos_tags,f)
f=open('pos_tags_test.pkl','wb')
pickle.dump(test_pos_tags,f)
