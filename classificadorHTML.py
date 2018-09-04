import urllib2
from bs4 import BeautifulSoup
from collections import Counter
from string import punctuation
import sys

# abre uma pagina web
response = urllib2.urlopen("http://jomi.das.ufsc.br")
# converte a pagina em texto
page_source = response.read()

#converte o texto da pagina em um analisador Soup
soup = BeautifulSoup(page_source, 'html.parser')

#imprime o texto do soup
#print(soup.get_text())

# Escolhe todo o texto ou filtra tags especificas
text = (''.join(s.findAll(text=True))for s in soup.findAll(True))
#text = (''.join(s.findAll(text=True))for s in soup.findAll('p'))
#text = (''.join(s.findAll(text=True))for s in soup.findAll(['title','p']))

# faz a contagem de palavras
c = Counter((x.rstrip(punctuation).lower() for y in text for x in y.split()))

# imprime as palavras mais comuns
palavras_comuns = c.most_common()
#print (palavras_comuns) # prints most common words staring at most common.

# imprime apenas as palavras com repeticao menor que a escolhida
repeticaoMinima = 10
palavrasComMinimoDeRepeticoes = [x for x in c if c.get(x) > repeticaoMinima]
#print (palavrasComMinimoDeRepeticoes) # words appearing more than 5 times

#print(sys.version)