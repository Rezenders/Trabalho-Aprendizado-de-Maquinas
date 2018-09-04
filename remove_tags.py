import sys
import re

#cleanr = re.compile('<.*?>') #Alternative regular expression
cleanr = re.compile(r'<[^>]+>')

# Removes all the html tags contained in the string passed
# @param raw_html string with a full html file, with all the tags
# @returns string without html tags
def remove_html_tags(raw_html):
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext

## The following code was used for testing and will not be used in the final code
# Kept for reference

if __name__ != "__main__":
    print("Not being executed as main")
    sys.exit(-1)

if len(sys.argv) < 2:
    print("Too few arguments!")
    print("Usage: file1")
    sys.exit(-1)

with open(sys.argv[1]) as open_file:
    #print(open_file.read())
    cleantext = remove_html_tags(open_file.read())
    print(cleantext)