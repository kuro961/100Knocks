import spacy

def join_token_text(sent):
    return ' '.join([token.text for token in sent])

def main():
    nlp = spacy.load('ja_core_news_md')
    for src, dst in [('kftt-data-1.0/data/orig/kyoto-train.ja', 'train_spacy.ja'),
                     ('kftt-data-1.0/data/orig/kyoto-dev.ja', 'dev_spacy.ja'),
                     ('kftt-data-1.0/data/orig/kyoto-test.ja', 'test_spacy.ja')]:
        with open(src) as s, open(dst, 'w') as d:
            for line in s:
                line = line.strip()
                doc = nlp(line)
                d.write(''.join([join_token_text(sent) for sent in doc.sents]) + '\n')

    nlp = spacy.load('en_core_web_md')
    for src, dst in [
        ('kftt-data-1.0/data/orig/kyoto-train.en', 'train_spacy.en'),
        ('kftt-data-1.0/data/orig/kyoto-dev.en', 'dev_spacy.en'),
        ('kftt-data-1.0/data/orig/kyoto-test.en', 'test_spacy.en'),
    ]:
        with open(src) as s, open(dst, 'w') as d:
            for line in s:
                line = line.strip()
                doc = nlp(line)
                d.write(''.join([join_token_text(sent) for sent in doc.sents]) + '\n')

if __name__ == '__main__':
    main()