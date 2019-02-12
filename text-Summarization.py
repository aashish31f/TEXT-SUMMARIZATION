# TEXT SUMMARIZATION

# Importing Libraries
import nltk 
from nltk.tokenize import sent_tokenize , word_tokenize

# Importing dataset
dataset = open('President-Speech.txt').read()

#Create the function

def summaryy(dataset):
    result = []
    for number,sentence in enumerate(sent_tokenize(dataset)):
        number_tokens = len(word_tokenize(sentence))
        tagged = nltk.pos_tag(word_tokenize(sentence))
        number_nouns = len([word for word , pos in tagged if pos in ["NN","NNP"]])
        ners = nltk.ne_chunk(nltk.pos_tag(word_tokenize(sentence)),binary = False)
        number_ners = len([chunk for chunk in ners if hasattr(chunk,'label')])
        score = (number_ners + number_nouns)/float(number_tokens)
        result.append((number,score,sentence))
    return result

summ = summaryy(dataset)

#Print the summary
for i in sorted(summ , key = lambda x: x[1],reverse = True):
    print(i[2])

        