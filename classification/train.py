import torch
import torch.nn as nn
from gensim.models import KeyedVectors
from classification.models import SubgroupClassifier

import math

torch.manual_seed(0)


def train(input, target):
    model.train()
    model.zero_grad()

    output = model(input)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    return loss.item()



if __name__ == "__main__":

    # Define the definitional words for race
    words = {
        'white': ["caucasian", "white", "europe", "european", "america", "american", "english", "england", "french", "france", "spanish", "spain", "italian", "italy", "australian", "australia", "canada", "canadian", "swedish", "sweden", "norway", "blond", "blonde", "britain", "russia", "russian"],
        'black': ["black", "africa", "african", "afro", "ghana", "ghanaian", "nigeria", "nigerian", "cameroon", "cameroonian", "kenya", "kenyan", "senegal", "senegalese", "zimbabwe", "zimbabwean", "zambia", "zambian"],
        'asian': ["asian", "asia", "oriental", "japan", "japanese", "china", "chinese", "korea", "korean", "indonesia", "indonesian", "thailand", "thai", "vietnam", "vietnamese", "india", "indian", "malaysia", "malaysian", "bangladesh", "bangladeshi", "bengali"],
        'hispanic': ["hispanic", "latino", "latina", "mexico", "mexican", "brazil", "brazilian", "argentina", "argentinian", "native", "chile", "chilean", "columbia", "columbian", "cuba", "cuban", "peru", "peruvian", "ecuador"],
        'arab': ["arab", "algeria", "algerian", "tunisia", "tunisian", "morocco", "moroccan", "egypt", "egyptian", "syria", "syrian", "yemen", "kuwait", "kuwaiti", "arabia", "arabian", "sudan", "sudanese", "lebanon", "lebanese", "saudi", "emirates", "qatar", "iraq", "syria", "jordan"],
        'other': []
    }


    # Read the most frequent words in english
    def_words = []
    for k in words.keys():
        def_words += words[k]

    with open("classification/data/most_frequent_words.txt", 'r') as f:
        words['other'] = [x.lower().strip() for x in f.readlines() if x.lower().strip() not in def_words]


    # Define the embeddings algorithm.  --- Glove in this case ---
    embedding_filepath = "embeddings/glove.txt"
    embedding_vectors = KeyedVectors.load_word2vec_format(embedding_filepath, binary=False)


    # Prepare the training data
    x_train = []
    y_train = []
    y_label = -1
    for race, def_words in words.items():
        y_label += 1
        for w in def_words:
            try:
                x_train.append(embedding_vectors[w])
                y_train.append(y_label)
            except:
                print("{} not in the vocabulary".format(w))
    

    # Transform into tensors
    x_train = torch.tensor(x_train)
    y_train = torch.tensor(y_train)


    # Shuffle the training data
    idx = torch.randperm(x_train.shape[0])
    x_train = x_train[idx]
    y_train = y_train[idx]


    # Define the model and hyperparameters
    hyperparameters = {
        "embedding_dim": 300,
        "num_classes": 6,
        "hidden_dim_1": 200,
        "hidden_dim_2": 100,
        "dropout": 0.3,
        "learning_rate": 0.0001,
        "minibatch_size": 100,
        "n_epochs": 500,
        "manual_seed": 0
    }

    model_filename = "classification/models/race_classifier.pt"
    weights = [len(words['white']), len(words['black']), len(words['asian']), len(words['hispanic']), len(words['arab']), len(words['other'])]
    min_weight = min(weights)
    weights = [min_weight/w for w in weights]
    weights = torch.tensor(weights)
    print(weights)

    model = SubgroupClassifier(embedding_dim=hyperparameters['embedding_dim'], num_classes=hyperparameters['num_classes'], hidden_dim_1=hyperparameters['hidden_dim_1'], hidden_dim_2=hyperparameters['hidden_dim_2'], dropout=hyperparameters['dropout'])
    criterion = nn.CrossEntropyLoss(reduction='mean', weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters['learning_rate'])


    # Start the training

    num_batches = math.ceil(x_train.shape[0] / hyperparameters['minibatch_size'])
    best_loss = float('inf')
    best_epoch = 0

    for epoch in range(hyperparameters['n_epochs']):
        train_loss = 0

        for i in range(num_batches):
            boundary = i * hyperparameters['minibatch_size']
            input = x_train[boundary: boundary + hyperparameters['minibatch_size']]
            target = y_train[boundary: boundary + hyperparameters['minibatch_size']]

            train_loss += train(input, target)


        if train_loss < best_loss:
            best_loss = train_loss
            best_epoch = epoch
            checkpoint = {
                'model': model.state_dict(),
                'hp': hyperparameters,
                'epoch': epoch
            }
            torch.save(checkpoint, model_filename)


        print("Training {:.2f}% --> Training Loss = {:.4f}".format(round(((epoch + 1) / hyperparameters['n_epochs']) * 100, 2), train_loss))

    
    print("######################")
    print("Race Training complete")
    print("######################")
    



































    # Define the definitional words for religion
    words = {
        'muslim': ["islam", "muslim", "muhammad", "sunni", "shia", "quran", "islamist", "mosque", "imam", "mecca", "medina", "islamic", "allah"],
        'christian': ["christianity", "christian", "church", "catholicism", "catholic", "jesus", "baptism", "orthodox", "christ", "testament", "bible", "baptist", "gospel", "messiah", "trinity", "apostle", "mary"],
        'jew': ["judaism", "jew", "jewish", "torah", "talmud", "synagogue", "hebrew", "moses", "israel", "temple", "jerusalem", "rabbi"],
        'other': []
    }


    # Read the most frequent words in english
    def_words = []
    for k in words.keys():
        def_words += words[k]

    with open("classification/data/most_frequent_words.txt", 'r') as f:
        words['other'] = [x.lower().strip() for x in f.readlines() if x.lower().strip() not in def_words]


    # Define the embeddings algorithm.  --- Glove in this case ---
    embedding_filepath = "embeddings/glove.txt"
    embedding_vectors = KeyedVectors.load_word2vec_format(embedding_filepath, binary=False)


    # Prepare the training data
    x_train = []
    y_train = []
    y_label = -1
    for race, def_words in words.items():
        y_label += 1
        for w in def_words:
            try:
                x_train.append(embedding_vectors[w])
                y_train.append(y_label)
            except:
                print("{} not in the vocabulary".format(w))
    

    # Transform into tensors
    x_train = torch.tensor(x_train)
    y_train = torch.tensor(y_train)


    # Shuffle the training data
    idx = torch.randperm(x_train.shape[0])
    x_train = x_train[idx]
    y_train = y_train[idx]


    # Define the model and hyperparameters
    hyperparameters = {
        "embedding_dim": 300,
        "num_classes": 4,
        "hidden_dim_1": 200,
        "hidden_dim_2": 100,
        "dropout": 0.3,
        "learning_rate": 0.0001,
        "minibatch_size": 100,
        "n_epochs": 500,
        "manual_seed": 0
    }

    model_filename = "classification/models/religion_classifier.pt"
    weights = [len(words['muslim']), len(words['christian']), len(words['jew']), len(words['other'])]
    min_weight = min(weights)
    weights = [min_weight/w for w in weights]
    weights = torch.tensor(weights)
    print(weights)

    model = SubgroupClassifier(embedding_dim=hyperparameters['embedding_dim'], num_classes=hyperparameters['num_classes'], hidden_dim_1=hyperparameters['hidden_dim_1'], hidden_dim_2=hyperparameters['hidden_dim_2'], dropout=hyperparameters['dropout'])
    criterion = nn.CrossEntropyLoss(reduction='mean', weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters['learning_rate'])


    # Start the training

    num_batches = math.ceil(x_train.shape[0] / hyperparameters['minibatch_size'])
    best_loss = float('inf')
    best_epoch = 0

    for epoch in range(hyperparameters['n_epochs']):
        train_loss = 0

        for i in range(num_batches):
            boundary = i * hyperparameters['minibatch_size']
            input = x_train[boundary: boundary + hyperparameters['minibatch_size']]
            target = y_train[boundary: boundary + hyperparameters['minibatch_size']]

            train_loss += train(input, target)


        if train_loss < best_loss:
            best_loss = train_loss
            best_epoch = epoch
            checkpoint = {
                'model': model.state_dict(),
                'hp': hyperparameters,
                'epoch': epoch
            }
            torch.save(checkpoint, model_filename)


        print("Training {:.2f}% --> Training Loss = {:.4f}".format(round(((epoch + 1) / hyperparameters['n_epochs']) * 100, 2), train_loss))

    
    print("######################")
    print("Religion Training complete")
    print("######################")


























     # Define the definitional words for gender
    words = {
        'man': ['male', 'he', 'him', 'his', 'himself', 'brother', 'father', 'grandfather', 'guy', 'husband', 'son', 'beau', 'boyfriend', 'man', 'boy', 'gentleman', 'nephew', 'papa', 'sir', 'uncle', 'daddy', 'men', 'dad', 'schoolboy', 'father', 'grandfathers', 'boys', 'schoolboys', 'gentlemen', 'husbands'],
        'woman': ['female', 'she', 'her', 'herself', 'sister', 'mother', 'grandmother', 'gal', 'wife', 'daughter', 'girlfriend', 'woman', 'girl', 'niece', 'mama', 'madam', 'aunt', 'women', 'mom', 'schoolgirl', 'mothers', 'grandmothers', 'girls', 'schoolgirls', 'ladies', 'lady', 'wives'],
        'other': []
    }


    # Read the most frequent words in english
    def_words = []
    for k in words.keys():
        def_words += words[k]

    with open("classification/data/most_frequent_words.txt", 'r') as f:
        words['other'] = [x.lower().strip() for x in f.readlines() if x.lower().strip() not in def_words]


    # Define the embeddings algorithm.  --- Glove in this case ---
    embedding_filepath = "embeddings/glove.txt"
    embedding_vectors = KeyedVectors.load_word2vec_format(embedding_filepath, binary=False)


    # Prepare the training data
    x_train = []
    y_train = []
    y_label = -1
    for race, def_words in words.items():
        y_label += 1
        for w in def_words:
            try:
                x_train.append(embedding_vectors[w])
                y_train.append(y_label)
            except:
                print("{} not in the vocabulary".format(w))
    

    # Transform into tensors
    x_train = torch.tensor(x_train)
    y_train = torch.tensor(y_train)


    # Shuffle the training data
    idx = torch.randperm(x_train.shape[0])
    x_train = x_train[idx]
    y_train = y_train[idx]


    # Define the model and hyperparameters
    hyperparameters = {
        "embedding_dim": 300,
        "num_classes": 3,
        "hidden_dim_1": 200,
        "hidden_dim_2": 100,
        "dropout": 0.3,
        "learning_rate": 0.0001,
        "minibatch_size": 100,
        "n_epochs": 500,
        "manual_seed": 0
    }

    model_filename = "classification/models/gender_classifier.pt"
    weights = [len(words['man']), len(words['woman']), len(words['other'])]
    min_weight = min(weights)
    weights = [min_weight/w for w in weights]
    weights = torch.tensor(weights)
    print(weights)

    model = SubgroupClassifier(embedding_dim=hyperparameters['embedding_dim'], num_classes=hyperparameters['num_classes'], hidden_dim_1=hyperparameters['hidden_dim_1'], hidden_dim_2=hyperparameters['hidden_dim_2'], dropout=hyperparameters['dropout'])
    criterion = nn.CrossEntropyLoss(reduction='mean', weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters['learning_rate'])


    # Start the training

    num_batches = math.ceil(x_train.shape[0] / hyperparameters['minibatch_size'])
    best_loss = float('inf')
    best_epoch = 0

    for epoch in range(hyperparameters['n_epochs']):
        train_loss = 0

        for i in range(num_batches):
            boundary = i * hyperparameters['minibatch_size']
            input = x_train[boundary: boundary + hyperparameters['minibatch_size']]
            target = y_train[boundary: boundary + hyperparameters['minibatch_size']]

            train_loss += train(input, target)


        if train_loss < best_loss:
            best_loss = train_loss
            best_epoch = epoch
            checkpoint = {
                'model': model.state_dict(),
                'hp': hyperparameters,
                'epoch': epoch
            }
            torch.save(checkpoint, model_filename)


        print("Training {:.2f}% --> Training Loss = {:.4f}".format(round(((epoch + 1) / hyperparameters['n_epochs']) * 100, 2), train_loss))

    
    print("######################")
    print("Gender Training complete")
    print("######################")