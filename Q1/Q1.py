import pandas as pd
import numpy as np
import os
import json
import string
import pickle
import math
import random
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords



class NaiveBayes:
    def __init__(self, path_train, path_test):
        print("Loading training data...")
        if not os.path.isfile("NB_simple_train.data"):
            self.trainX, self.trainY = self.load_clean(path_train, "NB_simple_train.data")
        else:
            self.trainX, self.trainY = pickle.load(open("NB_simple_train.data", "rb"))

        print("Training data loaded...")

        print("Loading test data...")

        if not os.path.isfile("NB_simple_test.data"):
            self.testX, self.testY = self.load_clean(path_test, "NB_simple_test.data")
        else:
            self.testX, self.testY = pickle.load(open("NB_simple_test.data", "rb"))

        print("Test data loaded...")

        print("Learning model...")
        if not os.path.isfile("simple_NB.model"):
            self.model = self.naiveBayes("simple_NB.model")
        else:
            self.model = pickle.load(open("simple_NB.model", "rb"))

        print("Model learned...")

        

    def load_clean(self, path, pickle_file):

        os.chdir(
            "C:/Users/Shubham/Desktop/Sem7/COL774_ML/Ass_2/Music_reviews_json/reviews_Digital_Music_5.json")

        X = []
        Y = []

        # print(os.getcwd())

        with open(path, "r") as f:
            for line in f:
                df = json.loads(line)
                X.append(df["reviewText"])
                Y.append(float(df["overall"]))

        os.chdir("C:/Users/Shubham/Desktop/Sem7/COL774_ML/Ass_2/Q1")

        # print("data loaded")


        # cleaning
        X = self.remove_punctuations_tokenize(X)


        # print(X[0])
        # print(Y[0])

        with open(pickle_file, 'wb') as file:
            pickle.dump([X, Y], file)

        return X, Y


    def remove_punctuations_tokenize(self, docs):

        answer = []
        # print(docs)

        for text in docs:
            data = ""
            for char in text:
                if(char in string.punctuation):
                    data += ' '
                elif(ord(char) >=65 and ord(char) <= 90):
                    data += chr(ord(char) + 32)
                else:
                    data += char


            #tokenize too
            data = data.split()
            answer.append(data)

        return answer

    def naiveBayes(self, pickle_file):
        # Using CountVectizer to fit nd transform data

        count_of_each_rating = [0, 0, 0, 0, 0]
        corr_text_list = [[], [], [], [], []]

        "print(type(vectorizer.vocabulary_))"
        # print(vectorizer.vocabulary_)

        total_training_cases = len(self.trainY)

        for i in range(total_training_cases):
            count_of_each_rating[int(self.trainY[i]) - 1] += 1
            corr_text_list[int(self.trainY[i]) - 1].append(self.trainX[i])


        global_vocabulary = dict()

        vocab_for_each_label = [dict(), dict(), dict(), dict(), dict()]

        for i in range(5):
            for text in corr_text_list[i]:
                for word in text:
                    if word not in global_vocabulary:
                        global_vocabulary[word] = 1
                    else:
                        global_vocabulary[word] += 1

                    if word not in vocab_for_each_label[i]:
                        vocab_for_each_label[i][word] = 1
                    else:
                        vocab_for_each_label[i][word] += 1

        # P(label = ?)
        # Prior probabilities
        prob_of_labels = [0.0, 0.0, 0.0, 0.0, 0.0]

        prob_of_labels[0] = count_of_each_rating[0]/total_training_cases
        prob_of_labels[1] = count_of_each_rating[1]/total_training_cases
        prob_of_labels[2] = count_of_each_rating[2]/total_training_cases
        prob_of_labels[3] = count_of_each_rating[3]/total_training_cases
        prob_of_labels[4] = count_of_each_rating[4]/total_training_cases


        # For majority prediction in part B
        majority_class = 1.0
        temp = -1
        for i in range(5):
            if(count_of_each_rating[i] > temp):
                temp = count_of_each_rating[i]
                majority_class = float(i+1)


        # print(prob_of_labels)

        # P(x/y)


        total_words_each_label = []

        for i in range(5):
            total_words_each_label.append(sum(vocab_for_each_label[i].values()))


        # countList_1 is, phi_x/y
        # count_list contains no of data with corr label
        total_unique_words = len(global_vocabulary)

        for i in range(5):
            for j in vocab_for_each_label[i]:
                vocab_for_each_label[i][j] = (vocab_for_each_label[i][j] + 1) / (total_words_each_label[i] + total_unique_words)

        LR_Model = [prob_of_labels, vocab_for_each_label, total_words_each_label, total_unique_words, majority_class]

        # file_name = 'simple_NB.model'

        with open(pickle_file, 'wb') as file:
            pickle.dump(LR_Model, file)

        return LR_Model


    def part1a(self):
        prior_probs = self.model[0]

        # theta_list [dict(), dict(), dict(), dict(), dict()], each dict has words as keys and prob as value
        theta_list = self.model[1]
        total_unique_words_each_label = self.model[2]
        total_unique_words = self.model[3]
        
        labels_prediction = self.predict(self.trainX, total_unique_words, theta_list, prior_probs)

        correct_predictions = 0

        for i in range(len(self.trainY)):
            # output_file.write(str(labels_prediction[i]))
            # output_file.write(" ")
            if(self.trainY[i] == labels_prediction[i]):
                correct_predictions += 1

        # output_file.write('\n')
        accuracy = correct_predictions/len(self.trainY)
        print("Accuracy on training data:", accuracy)




        # test data prediction
        self.test_data_prediction = self.predict(self.testX, total_unique_words, theta_list, prior_probs)

        correct_predictions = 0

        for i in range(len(self.testY)):
            # output_file.write(str(labels_prediction[i]))
            # output_file.write(" ")
            if(self.testY[i] == self.test_data_prediction[i]):
                correct_predictions += 1

        # output_file.write('\n')
        self.test_accuracy = correct_predictions/len(self.testY)
        print("Accuracy on test data:", self.test_accuracy)


    def pred_helper(self, prob_list):
        p = 1.0
        value = prob_list[0]
        for i in range(len(prob_list)):
            if prob_list[i] > value:
                value = prob_list[i]
                p = float(i) + 1.0

        return p


    def predict(self, text_tokenized, total_unique_words, theta_list, p_label):

        prediction = []

        for text in text_tokenized:
            req_prob = [math.log(p_label[0]), math.log(p_label[1]), math.log(
                p_label[2]), math.log(p_label[3]), math.log(p_label[4])]

            for word in text:
                if word in theta_list[0]:
                    req_prob[0] += math.log(theta_list[0][word])
                else:
                    req_prob[0] += math.log(1/total_unique_words)

                if word in theta_list[1]:
                    req_prob[1] += math.log(theta_list[1][word])
                else:
                    req_prob[1] += math.log(1/total_unique_words)

                if word in theta_list[2]:
                    req_prob[2] += math.log(theta_list[2][word])
                else:
                    req_prob[2] += math.log(1/total_unique_words)

                if word in theta_list[3]:
                    req_prob[3] += math.log(theta_list[3][word])
                else:
                    req_prob[3] += math.log(1/total_unique_words)

                if word in theta_list[4]:
                    req_prob[4] += math.log(theta_list[4][word])
                else:
                    req_prob[4] += math.log(1/total_unique_words)

            prediction.append(self.pred_helper(req_prob))

        return prediction


    def part1b(self):
        length = len(self.testY)

        random_pred = 0
        correct = 0

        for i in self.testY:
            random_pred = float(math.ceil((random.random()) * 5))

            if(i == random_pred):
                correct += 1

        random_accuracy = correct / length

        print("Test set accuracy by learned model:", self.test_accuracy)
        print("Test set accuracy by random prediction:", random_accuracy)

        # Majority prediction
        correct = 0
        maj_class = self.model[4]

        for i in self.testY:
            if(i == maj_class):
                correct += 1

        majority_accuracy = correct / length

        print("Test set accuracy by majority prediction:", majority_accuracy)


    def part1c(self):
        cf_mat = np.zeros([5, 5])

        for i in range(len(self.test_data_prediction)):
            cf_mat[int(self.testY[i])-1][int(self.test_data_prediction[i])-1] += 1

        classes = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.plot_confusion_matrix(cm=cf_mat,
                            normalize=False,
                            target_names=classes,
                            title="Confusion Matrix", filename = "1c.png")

        self.plot_confusion_matrix(cm=cf_mat,
                            normalize=True,
                            target_names=classes,
                            title="Confusion Matrix, Normalized", filename = "1c_normalized.png")


    def plot_confusion_matrix(self, cm,
                              target_names,
                              title='Confusion matrix',
                              cmap=None,
                              normalize=True, filename = None):

        import matplotlib.pyplot as plt
        import numpy as np
        import itertools

        accuracy = np.trace(cm) / float(np.sum(cm))
        misclass = 1 - accuracy

        if cmap is None:
            cmap = plt.get_cmap('Blues')

        plt.figure(figsize=(7, 6))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()

        if target_names is not None:
            tick_marks = np.arange(len(target_names))
            plt.xticks(tick_marks, target_names, rotation=45)
            plt.yticks(tick_marks, target_names)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        thresh = cm.max() / 1.5 if normalize else cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(
            accuracy, misclass))
        if(filename != None):
            plt.savefig(filename)
        plt.show()


    def part1d(self):
        pass


    def _stem(self, doc, p_stemmer, en_stop, return_tokens):
        tokens = word_tokenize(doc.lower())
        stopped_tokens = filter(lambda token: token not in en_stop, tokens)
        stemmed_tokens = map(lambda token: p_stemmer.stem(token), stopped_tokens)
        if not return_tokens:
            return ' '.join(stemmed_tokens)
        return list(stemmed_tokens)


    def getStemmedDocuments(self, docs, return_tokens=True):
        """
            Args:
                docs: str/list(str): document or list of documents that need to be processed
                return_tokens: bool: return a re-joined string or tokens
            Returns:
                str/list(str): processed document or list of processed documents
            Example: 
                new_text = "It is important to by very pythonly while you are pythoning with python.
                    All pythoners have pythoned poorly at least once."
                print(getStemmedDocuments(new_text))
            Reference: https://pythonprogramming.net/stemming-nltk-tutorial/
        """
        en_stop = set(stopwords.words('english'))
        p_stemmer = PorterStemmer()
        if isinstance(docs, list):
            output_docs = []
            for item in docs:
                output_docs.append(self._stem(item, p_stemmer, en_stop, return_tokens))
            return output_docs
        else:
            return self._stem(docs, p_stemmer, en_stop, return_tokens)
