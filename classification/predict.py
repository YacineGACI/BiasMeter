import torch
from gensim.models import KeyedVectors
from classification.models import SubgroupClassifier


if __name__ == "__main__":

    # Load the classification model
    model_filename = "classification/models/race_classifier.pt"
    checkpoint = torch.load(model_filename)
    hyperparameters = checkpoint['hp']
    model = SubgroupClassifier(embedding_dim=hyperparameters['embedding_dim'], num_classes=hyperparameters['num_classes'], hidden_dim_1=hyperparameters['hidden_dim_1'], hidden_dim_2=hyperparameters['hidden_dim_2'], dropout=hyperparameters['dropout'])
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # Define the embeddings algorithm.  --- Glove in this case ---
    embedding_filepath = "embeddings/glove.txt"
    embedding_vectors = KeyedVectors.load_word2vec_format(embedding_filepath, binary=False)


    # Predict
    classes = ["white", "black", "asian", "hispanic", "arab", "other"]
    test_words = ['arab', 'algerian', 'chinese', 'venezuela', 'white', 'black', 'african', 'english', 'thai', 'syrian', 'people', 'human', 'hello', 'yemen', 'morocco', 'iraq', 'iranian', 'bengali', 'iranians', 'ecuador', 'ukrainian', 'russian', 'jordan', 'syria']
    
    # classes = ['muslim', 'christian', 'jew', 'other']
    # test_words = ['muslim', 'christian', 'jew', 'hello', 'rabbi', 'imam', 'arab', 'church', 'american', 'brahmin']

    # classes = ['man', 'woman', 'other']
    # test_words = ['man', 'woman', 'schoolboy', 'mommy', 'mother', 'mothers']

    x_test = torch.tensor([embedding_vectors[w] for w in test_words])
    y_test = model(x_test)
    _, y_labels = torch.max(y_test, dim = 1)


    for i in range(len(test_words)):
        print(test_words[i], " --> ", classes[y_labels[i]])