import urllib2
from bs4 import BeautifulSoup
from collections import Counter
from string import punctuation

response = urllib2.urlopen("http://jomi.das.ufsc.br")
page_source = response.read()

soup = BeautifulSoup(page_source, 'html.parser')

#print(soup.get_text())

text = (''.join(s.findAll(text=True))for s in soup.findAll(True))
#text = (''.join(s.findAll(text=True))for s in soup.findAll('p'))
#text = (''.join(s.findAll(text=True))for s in soup.findAll(['title','p']))

c = Counter((x.rstrip(punctuation).lower() for y in text for x in y.split()))

palavras_comuns = c.most_common()
#print (palavras_comuns) # prints most common words staring at most common.

palavrasComMinimoDeRepeticoes = [x for x in c if c.get(x) > 10]
#print (palavrasComMinimoDeRepeticoes) # words appearing more than 5 times