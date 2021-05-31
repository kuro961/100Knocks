from prog20 import my_argparse, get_uk_text
import re

def main():
    args = my_argparse()

    uk_texts = get_uk_text(args.file)

    ans = [re.search(r'(?<=\[\[Category:)[^|*\]]+', i) for i in uk_texts]
    ans = [m_obj.group() for m_obj in ans if m_obj is not None]
    print(ans)

if __name__ == '__main__':
    main()