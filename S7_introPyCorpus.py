# -*- coding: utf-8 -*-
"""
practica de nltk 1, posibilidades del programa para uso con corpus
@weg
testado en Ipython w/ Mac
28/02/2018
"""
#pip install -U nltk

#vamos a usar este paquete, así que hay que importarlo antes, o no funcionarán 
#sus funciones
import nltk


#para ver todo el contenido que se puede bajar interactivamente
#nltk.download()
# y os mostrara todos los paquetes que hay



# EMPIEZA EL JUEGO
######

#declaro una variable con el nombre que yo quiera y le doy un contenido
miFrase = """At eight o'clock on Thursday morning Arthur didn't feel very good."""


#########################
# TOKENIZAR
####################
#me bajo el paquete que 
nltk.download('punkt')
misTokens = nltk.word_tokenize(miFrase)
print(misTokens)


############
# TAGGER AUTOMATICO DE U. PENN
############
nltk.download('averaged_perceptron_tagger')
miTagged = nltk.pos_tag(misTokens)
#nosenseña solo del primero al 6
miTagged[0:6]

#que significan las etiquetas
#nos lo podemos imaginar (apuestas?) pero vamos a preguntarselo
#nos bajamos las etiquetas
nltk.download('tagsets')
# nos enseña todas las etiquetas del corpus de upenn, podría ser de otro
nltk.help.upenn_tagset()
# nos da el significado de IN
nltk.help.upenn_tagset('IN')


#Vale, el tagger funciona, pero funciona bien?
# pero funciona bien? qué pasa si le damos una palabra ambigua como refuse
ambiguo = nltk.word_tokenize("They refuse to permit us to obtain the refuse permit") 
nltk.pos_tag(ambiguo) 


############
# etiquetas semánticas
############
#volvemos a usar la frase con la que hemos empezado
nltk.download('maxent_ne_chunker')
nltk.download('words')
misEntities = nltk.chunk.ne_chunk(miTagged)
print(misEntities)

nltk.download('treebank')
# veurem una llista dels fitxers dispobles, el 10% del world street journal
#treebank.fileids()
#escolliu el fitxer que volgue per fer el parsing
#jo agafo el primer, a py es comença a contar amb 0
miFraseEscogida = nltk.corpus.treebank.parsed_sents('wsj_0001.mrg')[0]
miFraseEscogida.draw()


#######
# vamos a sacar las concordancias de un corpus
#######
nltk.download('gutenberg')
import nltk.corpus
from nltk.text import Text
nltk.corpus.gutenberg.fileids()
#vamos a cargar un libro, pero podríais cargar cualquier fichero txt
#cualquier corpus que os bajéis de internet
moby = Text(nltk.corpus.gutenberg.words('melville-moby_dick.txt'))
moby.concordance('whale')

