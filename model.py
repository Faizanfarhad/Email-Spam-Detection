import numpy as np
import pandas as pd
import re
import nltk 
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split 
import pickle


df = pd.read_csv('/home/faizan/Documents/email spam-detaction/spam_ham_dataset.csv')
df = df.drop(['Unnamed: 0', 'label_num'],axis=1)
df['label'] = df['label'].str.contains('spam').astype(int)
text = df['text'].to_string()

def preprocessed_text(text):
    text = str(text)
    lowercase_word = [word.lower() for word in text.split()]
    removed_punctuation = re.sub(r"[^a-zA-Z0-9\s]", '', str(lowercase_word))
    removed_punctuation2 = re.sub(r"http\S+", "", removed_punctuation)  # remove links
    removed_punctuation3 = re.sub(r"[^a-zA-Z]", " ", removed_punctuation2)  # keep only letters
    tokens = nltk.word_tokenize(removed_punctuation3, language="english")
    stopword = set(stopwords.words("english"))
    tokens = [word for word in tokens  if word not in stopword ]
    lemitizer = WordNetLemmatizer()
    token_lemitize = [lemitizer.lemmatize(words) for words in tokens]
    return ' '.join(token_lemitize)

df['preprocessed_text'] = df['text'].apply(preprocessed_text)
df['text'] = df['preprocessed_text']

tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,
    min_df=2,
    max_df=0.85,
    ngram_range=(1,2),
    norm='l2',
    sublinear_tf=True)
tfidf_matrix = tfidf_vectorizer.fit_transform(df['preprocessed_text'])
tfidf_features = tfidf_vectorizer.get_feature_names_out()
tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=tfidf_features
)

X = tfidf_df
y = df['label']
X = np.array(X)
y = np.array(y)
x_train, x_test, y_train, y_test = train_test_split(X,y,train_size=0.8,random_state=42)
y_train = np.array(y_train)
y_test = np.array(y_test)
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression(x,w,b):
    z = np.dot(x,w) + b
    return sigmoid(z)

def compute_loss(m,prediction,y):
    epsilon = 1e-15
    cost = (-1/m) * np.sum(y * np.log(prediction + epsilon) + (1 - y) * np.log(1 -prediction + epsilon))
    return cost

def train(X:pd.DataFrame,y:pd.DataFrame,epoc=1500,learning_rate=0.01, batch_size=65):
    m,n = X.shape
    X = np.array(X)
    y = np.array(y)
    y = y.reshape(-1,1)
    w = np.zeros((n,1))
    b = 0
    for i in range(epoc):
        incides = np.arange(m)
        np.random.shuffle(incides)
        X = X[incides]
        y = y[incides]
        for i in range(0,m,batch_size):
            x_batch = X[i:i + batch_size]
            y_batch = y[i:i + batch_size].reshape(-1,1)
            
            z = np.dot(x_batch,w) + b
            prediction = sigmoid(z)
            dw = (1/x_batch.shape[0]) * np.dot(x_batch.T, (prediction - y_batch))
            db =  (1/x_batch.shape[0]) * np.sum(prediction - y_batch)

            w = w - learning_rate * dw
            b = b - learning_rate * db

            if i % 100 == 0:
                print(f'Prediction after {i} itreation : {prediction[1:8]}')


        if i % 100 == 0:
            cost = compute_loss(m,prediction,y_batch)
            print(f"Cost after {i} iteration : {cost}")
    print(f'weight after all the epoc : {w}\n bias after all the epoc : {b}')
    return w,b



if __name__ == '__main__':
    text =  " Your account has been locked.A8426887620Someone is attempting to sign-in to your account.When: Mon, June 10, 2024Device: Apple iPhone iOSNear: New Jersey, United StatesOur system automatically lock your account until your confirm this action.If this was you, click here to approve and the system will automatically unlock your account & you can sign-in again.If you didn’t request this sign-in: click here to deny and follow the instruction to unlock your account.Regards,Amazon Customer Service."
    w,b = train(x_train,y_train)


    with open("my_model.pkl", "wb") as f:
        pickle.dump((w, b), f)  # Save the model

    with open("my_vectorizer.pkl", "wb") as f:
        pickle.dump(tfidf_vectorizer, f)



    predictions = logistic_regression(x_test,w,b)
    pred = ['Spam' if pred > 0.5 else 'ham' for pred in predictions]
    y_actual = ['Spam' if i == 1 else 'ham' for i in y_train]
    # print(pred[1:40])
    final_pred = (predictions > 0.5).astype(int)
    accuracy = np.mean(final_pred == y_test) * 100
    # print(f"Accuracy: {accuracy:.2f}%")
    def custom_test(text):
        cleaned_text = preprocessed_text(text)
        vectorized = tfidf_vectorizer.transform([cleaned_text]).toarray()
        z = np.dot(vectorized, w) + b
        prob = sigmoid(z)
        label = 'Spam' if prob > 0.5 else 'Ham'
        return label,prob

    text_label,text_prob = custom_test(text)
    print(f'accuracy : {accuracy}')
    print(f"Prediction: {text_label}, Probability: {text_prob}")
        


'''
            spam_text2 =  "From:To:REQUEST A DEMOQdomain@domain-name.comSubject:Your emailApple Facetime Information DisclosureNational Security DepartmentA vulnerability has been identified in the Apple Facetime mobile applications that allow an attacker to record calls and videos from your mobile device without your knowledge.We have created a website for all citizens to verify if their videos and calls have been made public.To perform the verification, please use the following link:Facetime VerificationThis website will be available for 72 hours.National Security Department"
            label,prob = custom_test(spam_text)
            label2,prob2 = custom_test(spam_text2)

            ham_text = "Hi GHOST,Get ready to tune into Google I/O, where you can explore the latest innovations from Google! Join us online May 20-21 for live streamed keynotes and sessions covering what’s new in AI, Android, web, cloud, and more."
            ham_text1 = "Hello Faizan,Represent Internsha,at Chanderprabhu Jain(CPJ) College Nerala,Del,Join India's leading Student Partner Progr,We're inviting you to participate in our student program that offers earning potential up to ₹7 lakhs while gaining valuable career experience.Program highlights: Participants hav,opportunities to win rewards including an iPhone 14, cash prizes up to ₹10,000, Internshala merchandise,and complimentary training courses.Skill development: Gain practical experience in leadership, professional communication, an,networking through 15+ online activities.Inclusive opportunity: Available to all students regardless of college, academic stream, or educational background.Deadline: The application window closes tomorrow."
            ham_label,ham_prob = custom_test(ham_text)
            ham_label1,ham_prob1 = custom_test(ham_text1)

'''