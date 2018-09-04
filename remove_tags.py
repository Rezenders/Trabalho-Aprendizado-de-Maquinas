import sys
import re
from pathlib import Path
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

#cleanr = re.compile('<.*?>') #Alternative regular expression
cleanr = re.compile(r'<[^>]+>')

# Removes all the html tags contained in the string passed
# @param raw_html string with a full html file, with all the tags
# @returns string without html tags
def remove_html_tags(raw_html):
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext

# Removes all the html tags contained in the html file passed
# @param file_path path of a html file
# @returns string without html tags
def remove_tags_from_file(file_path):
    with open(Path(file_path)) as open_file:
        return remove_html_tags(open_file.read())

# Removes all the stopwords contained in the text passed
# @param clean_html text as string
# @returns string without stopwords
def remove_stop_words(clean_html):
    stop_words = set(stopwords.words('english'))

    # Add words that are not useful to the set
    stop_words.update(['MIME-Version','text/html','Content-Length','Content-Type'])
    
    word_tokens = word_tokenize(clean_html)
    
    clean_stop_words = [w for w in word_tokens if not w in stop_words]
    
    return clean_stop_words

def html_pre_processing(html_path):
    with open(Path(html_path)) as html_file:
        clean_html = remove_html_tags(html_file.read())
        
        clean_stop_words = remove_stop_words(clean_html)
        print(clean_stop_words)
        


## The following code was used for testing and will not be used in the final code
# Kept for reference

#if __name__ != "__main__":
#    print("Not being executed as main")
#    sys.exit(-1)
#
#if len(sys.argv) < 2:
#    print("Too few arguments!")
#    print("Usage: file1")
#    sys.exit(-1)
#
#with open(sys.argv[1]) as open_file:
#    #print(open_file.read())
#    cleantext = remove_html_tags(open_file.read())
#    print(cleantext)
