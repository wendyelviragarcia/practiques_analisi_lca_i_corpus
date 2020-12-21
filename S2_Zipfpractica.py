# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 12:31:24 2018

@author: labfonub15
"""
import nltk
#import re
import numpy
from operator import itemgetter



######
#     
#   CARREGUEM EL LLIBRE QUE HEM ESCOLLIT
#
######
#input = f_open = open('D:\\Usuarios\\labfonub15\\Desktop\\lotr.txt', 'r')
#input = f_open = open('/Users/weg/Desktop/lotr2.txt', 'r')
import codecs
input = codecs.open('/Users/weg/Desktop/lotr.txt', encoding='utf-8')
contenido = input.read()
input.close()


######
# ara a contenido tenim el contingut del llibre anem a explorar com 
#es, a veure si te un tipus que podem tokenitzar



tokens = nltk.word_tokenize(contenido)
type(tokens)
len(tokens)
tokens[:10]


#ara treurem les frequencies de les paraules
# sabeu que es un bucle?
frequency={} #fem un diccionari, encara buit
for word in tokens:
    count = frequency.get(word,0)
    frequency[word] = count + 1

frequency['Gandalf']


frequency['Bag']
frequency['bag']

# aquesta és la frec absoluta i la relativa? frecuencia per 1 milio de praules
# quantes paraules té el meu llibre
len(tokens)

#llavors la freq relativa de Gandalf sera
# frecuencia de Gandalf entre total del meu corpus per un milio de paraules
freqRelativaGandalf = ((frequency['Gandalf'])/len(tokens))*1000000
#i si vull el percetatge? doncs, la meva frequencia entre el total del corpus i multiplicat per 100
freqPercetageGandalf = ((frequency['Gandalf'])/len(tokens))*100
# el 0.2% del corpus es gandalf


##############
#
# Ara farem un gràfic de les frequencies
##############
import matplotlib.pyplot as plt

#hem d'ordenar les dades per frequencia
ordenado = sorted(frequency.items(), key = itemgetter(1), reverse=True) # sorted by frec, return a list of tuples
itemsOrdenados, frecuencia = zip(*ordenado) # unpack a list of pairs into two tuples
rank = numpy.arange(1., 50.)
#el veiem sencer, no es veu res
plt.plot(frecuencia)

#anem a veure les 50 paraules més comuns

plt.plot(frecuencia[0:50], 'o-')
plt.show()





#pero aixo es frecuencia absoluta no podem comparar amb res
arr_freq= numpy.asarray(frecuencia, dtype=numpy.float32) #teniem un tuple hem de convertir les dades
#en un array de nombres per tal de poder dividir-les
plt.plot(arr_freq[0:50]/max(arr_freq), 'o-')
plt.show()


#voleu saber quines son?
plt.plot(arr_freq[0:50], 'o-')
plt.xticks(rank, itemsOrdenados, rotation='vertical')
plt.show()



#ara anem a veure la zipf normativa
from scipy import special
a= 2
y = rank**(-a) / special.zetac(a)
plt.plot(rank, y/max(y), linewidth=2, color='r')
plt.show()


#com les posem juntes? hem de fer-ho amb la frequencia normalitzada, 
# perquè tinguin l'eix igual de 0 a 1
#es compleix la llei perfectament?
plt.plot(arr_freq[0:50]/max(frecuencia), 'o-')
y = rank**(-a) / special.zetac(a)
plt.plot(rank, y/max(y), linewidth=2, color='r')
plt.xticks(rank, itemsOrdenados, rotation='vertical')
plt.show()




##############
#
# vamos a sacar las collocations (el otro día vimos concordancias)
# aquí podéis ver las medidas de asociacion y más cosas que se pueden hacer
# http://www.nltk.org/howto/collocations.html

###########
# para poder pasar la funcion necesitamos cambiar el tipo
import nltk.collocations
tokens=list(tokens)
texto = nltk.Text(tokens)
type(texto)
texto.collocations()


#en mi caso
#said Frodo; Bag End; said Gandalf; Minas Tirith; long ago; said
#Aragorn; Black Riders; Tom Bombadil; Misty Mountains; said Merry;
#could see; Elder Days; Mr. Butterbur; Great River; far away; said
#Pippin; said Strider; said Sam; either side; Dark Lord

#podemos ver las concordancias de "Bag end", por ejemplo, para ver 
#si solo aparece en ese contexto
texto.concordance('bag')

# frecuencia de bag? la teníamos antes, solo hace falta que la llamemos
     
###################################
#
# ideas de PREprocesamiento
#
###################################

# pasalo todo a minusculas así haras que Bag y bag sean la misma palabra

textoMinusculas = contenido.lower()
import string
for punct in string.punctuation:
    textoSinPuntuacion = contenido.replace(punct," ")
#los stopwords son opalabras frecuentes que no nos interesan para el procesado
from nltk.corpus import stopwords
sinStop = [w for w in tokens if not w in stopwords.words('english')]

 