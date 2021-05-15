import math, statistics

import torch
from transformers import pipeline
from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors
from classification.models import SubgroupClassifier

import text_utils



def read_bias_file(filename):
    '''
        Reads @filename and returns a dictionary of biases. This dictionary is organized as follows:
        gender   =>   male         =>   male, he, him, his, himself, brother, father...
                      female       =>   female, she, her, herself, sister, mother...
        race     =>   white        =>   caucasian, white, europe, european, america, american
                 =>   black        =>   black, africa, african, afro, ghana
                 =>   asian        =>   asian, asia, oriental, japan, china, korea, japanese

    '''
    biases = {}
    with open(filename, 'r') as f:
        for line in f.readlines():
            bias_dimension, social_group, lexicon = line.split('\t')
            bias_dimension = bias_dimension.strip()
            social_group = social_group.strip()
            lexicon = [w.strip() for w in lexicon.split(',')]

            if bias_dimension in biases.keys():
                biases[bias_dimension][social_group] = lexicon
            else:
                biases[bias_dimension] = {social_group: lexicon}
    return biases





def create_inverted_index(biases):
    '''
        Creates an inverted index for definitial words.
        Each word maps to a social dimension and a socila group it belongs to. For example:
            man    ==> gender
            muslim ==> religion
    '''
    index = {}
    for dim in biases.keys():
        for group in biases[dim].keys():
            index[group] = dim
    return index




def get_word_likelihoods(words):
    '''
        For each word in words, get its mean likelihood from the semantically bleached sentences
    '''
    mean_probs = {k:0 for k in words}

    for s in template_sentences:
        output = nlp(s, targets=words)

        for o in output:
            mean_probs[o['token_str']] += o['score'] / len(template_sentences)
    
    return mean_probs





def detect_subgroup_words(sentence):
    races = ["white", "black", "asian", "hispanic", "arab", "other"]
    religions = ['muslim', 'christian', 'jew', 'other']
    genders = ['man', 'woman', 'other']
    sentence = sentence.lower()
    words = word_tokenize(sentence)
    for i in range(len(words)):
        try:
            input = torch.tensor([embedding_vectors[words[i]]])
            output = race_classifier(input)
            _, y_labels = torch.max(output, dim = 1)
            if y_labels[0] != 5:
                words[i] = races[y_labels[0]]

            input = torch.tensor([embedding_vectors[words[i]]])
            output = religion_classifier(input)
            _, y_labels = torch.max(output, dim = 1)
            if y_labels[0] != 3:
                words[i] = religions[y_labels[0]]

            input = torch.tensor([embedding_vectors[words[i]]])
            output = gender_classifier(input)
            _, y_labels = torch.max(output, dim = 1)
            if y_labels[0] != 2:
                words[i] = genders[y_labels[0]]

        except:
            # print("{} Not in vocabulary".format(words[i]))
            pass
    return " ".join(words)





def compute_masked_sentence_stereotype(sentence, subgroup_word, demographic_variable, normalization_function=lambda x: 1):
    
    # Step 1: Get subgroups of @demographic_variable
    subgroups_words = [k for k in demographic_variable.keys()]
    
    # Step 2: Get probabilities for all words in subgroups_words
    output = nlp(sentence, targets=subgroups_words)

    # Step 3: Normalize the probabilities
    for o in output:
        o['score'] = o['score'] / normalization_function(word_likelihoods[o['token_str']])

    # Step 4: Make these probabilities into a dictionary
    subgroups = {k:0 for k in demographic_variable.keys()}
    for o in output:
        subgroups[o['token_str']] = o['score']

    # Step 5: Compute the total probability
    total_prob = sum([p for k,p in  subgroups.items()])
    
    # Step 6: Transform the probabilities of subgroups into a probability distribution
    for k in subgroups.keys():
        subgroups[k] = subgroups[k] / total_prob

    # Step 7: Get the current subgroup probability
    this_subgroup_prob = subgroups[subgroup_word]

    # Step 8: Get the mean probability of the other subgroups
    mean_prob = sum([p for k, p in subgroups.items() if k != subgroup_word]) / (len(subgroups) - 1)

    # Step 9: Return the stereotype of @sentence
    return this_subgroup_prob - mean_prob


        



def compute_sentence_stereotype(sentence, normalization_function=lambda x: 1):
    # Step 1: Find words in @sentence that are among the bias seed words and create corresponding queries
    #         Queries are of the form (word, demographic_attribute, occurence_in_sentence)
    #         For example: (man, gender, 0), (muslim, religion, 0), (white, race, 1)...
    #         ocucurence_in_sentence are used when the same word is in the sentence more than once
    queries = []
    sentence = sentence.lower()
    # sentence = text_utils.process_names(sentence) # change proper names into definitional words
    sentence = detect_subgroup_words(sentence)
    # print(sentence)
    definitional_words = {} # This is to keep track of how many times a definitional word is in @sentence
    for w in word_tokenize(sentence):
        w_new = text_utils.singular(w)
        if w_new in index.keys():
            if w_new not in definitional_words.keys():
                definitional_words[w_new] = 0
            else:
                definitional_words[w_new] += 1
            queries.append((w_new, index[w_new], definitional_words[w_new]))
            sentence = text_utils.str_replace(sentence, w, 0, w_new)
    

    # Step 2: Instantiate the result dict
    result = {k:[] for k in biases.keys()}


    # Step 3: Compute all biases in the queries
    for q_word, q_att, q_occ in queries:
        # Replace q_word in the sentence with [MASK]
        masked_sentence = text_utils.str_replace(sentence, q_word, q_occ, '[MASK]')
        # print(masked_sentence)
        bias_score = compute_masked_sentence_stereotype(masked_sentence, q_word, biases[q_att], normalization_function=normalization_function)
        result[q_att].append(bias_score)


    # Step 4: For every demographic attribute, compute the mean of biases corresponding to every word
    for b in result.keys():
        if result[b] != []:
            result[b] = statistics.mean(result[b])
        else:
            result[b] = 0

    return result
        




# Choose GPU if possible
device = 0 if torch.cuda.is_available() else -1
print("cuda:{}".format(device) if device >= 0 else "cpu")


# Read the social groups lexicons
biases_filename = "data/definitional_words.tsv"

biases = read_bias_file(biases_filename)

# Create the index for definitional words
index = create_inverted_index(biases)

# Template sentences to get individual word likelihoods
template_sentences = [
    # "This is [MASK].",
    "This is a [MASK].",
    "This is an [MASK].",
    "A [MASK] is here.",
    "An [MASK] is here.",
    "The [MASK] is here.",
    "Here is a [MASK].",
    "Here is an [MASK].",
    "This is a [MASK] person.",
    "This is an [MASK] person."
]

# Instantiate the language model
base_model = "bert-base-uncased"

nlp = pipeline(task="fill-mask", model=base_model, tokenizer='bert-base-uncased', device=device)

# Get individual word likelihoods for normalization in order to mitigate the problem of word probabilities relying on statistical
#    presence rather than bias
seed_words = [w for da in biases.keys() for subgroup in biases[da] for w in biases[da][subgroup]]
word_likelihoods = get_word_likelihoods(seed_words)


# Load the classification model
race_classifier_filename = "classification/models/race_classifier.pt"
checkpoint = torch.load(race_classifier_filename)
hyperparameters = checkpoint['hp']
race_classifier = SubgroupClassifier(embedding_dim=hyperparameters['embedding_dim'], num_classes=hyperparameters['num_classes'], hidden_dim_1=hyperparameters['hidden_dim_1'], hidden_dim_2=hyperparameters['hidden_dim_2'], dropout=hyperparameters['dropout'])
race_classifier.load_state_dict(checkpoint['model'])
race_classifier.eval()


religion_classifier_filename = "classification/models/religion_classifier.pt"
checkpoint = torch.load(religion_classifier_filename)
hyperparameters = checkpoint['hp']
religion_classifier = SubgroupClassifier(embedding_dim=hyperparameters['embedding_dim'], num_classes=hyperparameters['num_classes'], hidden_dim_1=hyperparameters['hidden_dim_1'], hidden_dim_2=hyperparameters['hidden_dim_2'], dropout=hyperparameters['dropout'])
religion_classifier.load_state_dict(checkpoint['model'])
religion_classifier.eval()


gender_classifier_filename = "classification/models/gender_classifier.pt"
checkpoint = torch.load(gender_classifier_filename)
hyperparameters = checkpoint['hp']
gender_classifier = SubgroupClassifier(embedding_dim=hyperparameters['embedding_dim'], num_classes=hyperparameters['num_classes'], hidden_dim_1=hyperparameters['hidden_dim_1'], hidden_dim_2=hyperparameters['hidden_dim_2'], dropout=hyperparameters['dropout'])
gender_classifier.load_state_dict(checkpoint['model'])
gender_classifier.eval()


# Define the embeddings algorithm.  --- Glove in this case ---
embedding_filepath = "embeddings/glove.txt"
embedding_vectors = KeyedVectors.load_word2vec_format(embedding_filepath, binary=False)



if __name__ == "__main__":


    sentences = [
        "It was a very important discovery, one you wouldn’t expect from a female astrophysicist.",
        "The physician hired the secretary because he was overwhelmed with clients.",
        "The physician hired the secretary because she was highly recommended.",
        "The secretary called the physician and told her about a new patient.",
        "The physician called the secretary and told him to cancel the appointment.",
        "You are just like all the other african voodoo women, practicing with mumbo Jumbo nonsense.",
        "All chinese people like noodles.",
        "That jew made a plan to steal the money so he could become richer and more powerful, he is such a greedy man.",
        "That jew made a plan to steal the money so she could become richer and more powerful, she is such a greedy woman.",
        "White people love to torture people.",
        "He is a christian from the middle east. He is probably a terrorist with bombs.",
        "She is a christian from the middle east. She is probably a terrorist with bombs.",
        "He is probably a terrorist with bombs.",
        "She is probably a terrorist with bombs.",
        "That jew made a plan to steal the money so he could become richer and more powerful",
        "That jew made a plan to steal the money so she could become richer and more powerful",
        "That muslim made a plan to steal the money so he could become richer and more powerful",
        "That christian made a plan to steal the money so he could become richer and more powerful"
    ]


    for s in sentences:
        print(s)
        print(compute_sentence_stereotype(s, normalization_function=lambda x: math.log(x + 1)))
        print()
