from prog50 import my_argparse
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocessing(text):
    text = text.lower()
    text = re.sub('[0-9]+', '0', text)

    return text

def main():
    args = my_argparse()

    train = pd.read_table(args.train)
    valid = pd.read_table(args.valid)
    test = pd.read_table(args.test)

    df = pd.concat([train, valid, test])

    df['TITLE'] = df['TITLE'].map(lambda x: preprocessing(x))

    train_valid = df[:len(train) + len(valid)]
    test = df[len(train) + len(valid):]

    vectorizer = TfidfVectorizer()

    x_train_vec = vectorizer.fit_transform(train_valid['TITLE'])
    x_test = vectorizer.transform(test['TITLE'])

    x_train_vec = pd.DataFrame(x_train_vec.toarray(), columns=vectorizer.get_feature_names())
    x_test = pd.DataFrame(x_test.toarray(), columns=vectorizer.get_feature_names())

    x_train = x_train_vec[:len(train)]
    x_valid = x_train_vec[len(train):]

    x_train.to_csv(args.train_f, sep='\t', index=False)
    x_valid.to_csv(args.valid_f, sep='\t', index=False)
    x_test.to_csv(args.test_f, sep='\t', index=False)

if __name__ == '__main__':
    main()