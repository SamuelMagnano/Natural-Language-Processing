# Natural-Language-Processing
Collection of the main exercises requested for the NLP course i attended and completed. 
There were several others needed to pass the exam but they were trivial as simply requested to test functionalities, therefore i will not upload them.

## N-Grams generation
Requests:
1. Find two different sets of tweets and then, for each such set, you will: 
	- acquire two language models (one for bi-grams and one for tri-grams);
	- employ these language models to generate text and compare the different styles
2. Train a bi-gram and tri-gram model from the epic-novel MobyDick and use it to generate brief texts

## 2. Document Classification using Rocchio Method
Request: 
- Preprocessing
- Clean and normalize text (lowercase, punctuation removal, etc.), tokenize and remove stopwords
- Use TfidfVectorizer from scikit-learn to convert documents to TFIDF vectors
- Split data into training and test sets 
- For each class compute the centroid based on the set of training documents for that class 
- For each test document compute cosine similarity to all class centroids 
- Assign the label of the closest centroid

Rocchio Formula:
- i-th word in j-th document weight:

$$
w_{ij} = tf_{ij} * idf_{ij}
$$

- Centroid

$$
\vec{c_i} = <f_{1i},...,f_{|\tau|i}>
$$

- i-th centroid's k-th feature

$$
f_{ki} = \beta\sum_{\vec{d}_j \in POS_i} \frac{w_{kj}}{|POS_i|}- \gamma \sum_{\vec{d}_j \in NEG_i} \frac{w_{kj}}{|NEG_i|}
$$

where:

$$
POS = \{ d_j \in Documents | \phi(d_j,c_i) = True\}
$$

$$
NEG = \{ d_j \in Documents  | \phi(d_j,c_i) = False\}
$$

## 3. Topic Modeling and Document Clustering

Documents clustering, using their embeddings, done following these steps:

- Convert the documents to embeddings (GTE)
- Embeddings dimensionality reduction (Umap)
- Clustering starting from the reduced embeddings (HDBSCAN)

Topic modeling done using BERTopic

## 4. LLM Prompting
LLM Prompting used to define labels starting from the topics found in repository 3 and definition-to-object starting from the definitions given for another exercise. 
To achieve the results i tried applying:

- Zero shot
- Few shot
- Chain of thoughts

## 5. Disambiguation given N-languages
The main object was the creation of an interlingua used to disambiguate the meanings associated to words in several languages.
To do so we chose the following languages and accessed their synsets via BabelNet:

- English
- Italian
- French
- Spanish
- Portugese
- Romanian

For each word in each language we obtain the related synsetsIDs and, starting from their intersection, we evaluare the amount of ambiguity reduction using the following formula:

$$
\text{AmbiguityReduction} = \frac{\sum_{i=1}^N |\mathcal{S}_i| - N \cdot \left|\bigcap_{i=1}^N \mathcal{S}_i\right|}{\sum_{i=1}^N |\mathcal{S}_i|}
$$

where:

$$ 
|\bigcap_{i=1}^N \mathcal{S}_i| \cdot N 
$$

are the senses shared amongst languages and

$$ 
\sum_{i=1}^N |\mathcal{S}_i| 
$$

is the totality of senses amongst the languages.
