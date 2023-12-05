import pandas as pd
import streamlit as st

import numpy as np
import nltk


#Titulo
st.write("""
Prevendo sentimento de mensagens\n
App que utiliza ML para prever o sentimento da mensagem enviada.\n
""")

#dataset
basetreinamento = [
('O sol da manhã ilumina meu dia e meu coração','Alegria'),
('Encontrar velhos amigos é como descobrir um tesouro perdido','Alegria'),
('O sorriso radiante de uma criança é um presente para a alma','Alegria'),
('Notícias inesperadamente boas fazem meu coração dançar','Alegria'),
('Rir juntos, a melhor música da vida','Alegria'),
('Alcançar metas traz uma satisfação indescritível','Alegria'),
('Dançar sem se importar com o mundo ao redor é libertador','Alegria'),
('O desabrochar de uma flor é um espetáculo encantador','Alegria'),
('Compartilhar refeições deliciosas é um banquete de emoções','Alegria'),
('Vibrações positivas ao ouvir uma música favorita','Alegria'),
('Descobrir algo novo e apaixonante é uma jornada inspiradora','Alegria'),
('Elogios sinceros são como raios de sol para a alma','Alegria'),
('Fazer o bem traz uma alegria duradoura','Alegria'),
('O cheiro de chuva renova as energias e o espírito','Alegria'),
('Compartilhar momentos à mesa com entes queridos é uma festa','Alegria'),
('Observar animais brincando traz sorrisos incontáveis','Alegria'),
('Vencer desafios é uma dança com a realização','Alegria'),
('Desfrutar de um dia sem pressa é um deleite','Alegria'),
('Abraços apertados são abraços de amor e alegria','Alegria'),
('Testemunhar gentilezas entre estranhos aquece o coração','Alegria'),
('O céu estrelado à noite é uma tapeçaria de felicidade','Alegria'),
('Uma piada que faz gargalhar é uma sinfonia de risos','Alegria'),
('Perder-se nas páginas de um livro é uma viagem inigualável','Alegria'),
('Um amanhecer sereno é um quadro de paz','Alegria'),
('Completar projetos traz uma sensação de realização','Alegria'),
('A luz suave das velas cria um ambiente mágico','Alegria'),
('Caminhar na natureza é uma terapia para a alma','Alegria'),
('Superar medos é abrir as asas da liberdade','Alegria'),
('Rir com amigos é uma experiência terapêutica','Alegria'),
('Surpreender-se com algo inesperado é como magia','Alegria'),
('Aprender algo novo é alimentar a curiosidade e alegria','Alegria'),
('Compartilhar o sucesso alheio é celebrar juntos','Alegria'),
('Desfrutar de um dia preguiçoso é uma pausa revigorante','Alegria'),
('Um amanhecer vibrante é uma explosão de cores e alegria','Alegria'),
('Criar algo com as próprias mãos é um ato de alegria','Alegria'),
('Receber uma carta de alguém querido é um presente','Alegria'),
('Desfrutar de uma xícara de café perfeita é um ritual de alegria','Alegria'),
('Uma conversa leve com um amigo é um bálsamo','Alegria'),
('Viajar para novos lugares é uma aventura repleta de alegria','Alegria'),
('Alcançar metas pessoais é uma jornada de celebração','Alegria'),
('Recordar memórias felizes é revisitar a alegria','Alegria'),
('Ter um dia produtivo é um ciclo de satisfação','Alegria'),
('Observar a natureza é uma fonte contínua de inspiração','Alegria'),
('Ser grato pelas pequenas coisas é um exercício diário de alegria','Alegria'),
('Ajudar alguém em necessidade é um ato de alegria pura','Alegria'),
('Assistir a um filme engraçado é uma explosão de risos','Alegria'),
('Reconectar-se com velhos amigos é um tesouro reencontrado','Alegria'),
('Ter um dia sem obrigações é um presente valioso','Alegria'),
('Rever um ente querido é um reencontro cheio de alegria','Alegria'),
('Dançar como se ninguém estivesse olhando é uma liberdade contagiante','Alegria'),
('Receber elogios pelo trabalho árduo é um reconhecimento de alegria','Alegria'),
('Experimentar algo novo e excitante é uma jornada eletrizante','Alegria'),
('Um momento espontâneo e inesperado é uma pérola de alegria','Alegria'),
('Ver a beleza nas coisas simples é um olhar encantado de alegria','Alegria'),
('Ter um dia cheio de risadas é um dia verdadeiramente memorável','Alegria'),
('Perder-se em uma paisagem deslumbrante é uma fuga revigorante','Alegria'),

('Teus olhos revelam um universo de carinho','Amor'),
('A sintonia entre nossos corações é uma sinfonia eterna','Amor'),
('Seu toque é a melodia que acalma minha alma','Amor'),
('Cada suspiro meu é uma prece pelo teu sorriso','Amor'),
('Em teus abraços, encontro o meu lar verdadeiro','Amor'),
('Tu és a poesia que dá vida às palavras do meu coração','Amor'),
('Nos teus gestos, encontro a ternura que me completa','Amor'),
('Nossas almas dançam juntas uma dança intemporal','Amor'),
('Teus beijos são a magia que colore meus dias','Amor'),
('A luz do teu olhar é meu farol na escuridão','Amor'),
('Nossas risadas constroem pontes de cumplicidade','Amor'),
('Tu és o sol que ilumina meu universo particular','Amor'),
('Em teu silêncio, ouço as palavras não ditas de carinho','Amor'),
('Tua presença é o abraço que aquece minha alma','Amor'),
('Nossos sonhos entrelaçados formam um destino único','Amor'),
('Tu és o jardim secreto onde floresce nosso afeto','Amor'),
('A batida do teu coração é a música da minha existência','Amor'),
('Teus segredos são pérolas guardadas em meu coração','Amor'),
('Tu és a estrela que guia meus passos na escuridão','Amor'),
('A ternura do teu olhar acalenta as tormentas em mim','Amor'),
('Em teus lábios, descubro o sabor dos nossos dias felizes','Amor'),
('Nossa história é um livro escrito pelos nossos sorrisos','Amor'),
('Tu és o refúgio onde encontro paz nos dias turbulentos','Amor'),
('Nossos abraços são laços que nunca se desfazem','Amor'),
('A história dos nossos dias se tece nos fios do carinho','Amor'),
('Em teus sonhos, encontro um universo de possibilidades','Amor'),
('Teu riso é a canção que embala minhas noites solitárias','Amor'),
('Nossa jornada é um poema escrito nas páginas do tempo','Amor'),
('Cada palavra tua é um verso que eternizo no meu ser','Amor'),
('Nosso amor é um quadro onde as cores dançam em harmonia','Amor'),
('Tua presença é o farol que guia meu barco nas tempestades','Amor'),
('A verdade do teu olhar é a bússola do meu destino','Amor'),
('Tu és o eco das minhas palavras sussurradas ao vento','Amor'),
('A ternura do teu toque é a carícia que me envolve','Amor'),
('Nossos olhares contam histórias sem precisar de palavras','Amor'),
('Tua voz é a melodia que embala os sonhos do meu coração','Amor'),
('A chama do teu amor aquece minhas noites mais frias','Amor'),
('Nosso compromisso é escrito nas estrelas do céu noturno','Amor'),
('Tu és a chave que abre as portas do meu ser mais profundo','Amor'),
('A harmonia dos nossos silêncios é a linguagem do coração','Amor'),
('Nossas promessas são feitas nas estrelas, eternas e brilhantes','Amor'),
('Tu és o arco-íris após a tempestade na minha jornada','Amor'),
('Nosso amor é a dança eterna das almas apaixonadas','Amor'),
('Tua presença é o presente que enfeita o meu amanhã','Amor'),
('Cada lágrima tua é um oceano que navego com cuidado','Amor'),
('Nosso amor é a história que se desenha a cada nascer do sol','Amor'),
('Em teus sonhos, encontro moradas de afeto e esperança','Amor'),
('Tua risada é a melodia que embala minhas noites solitárias','Amor'),
('Nossa cumplicidade é o segredo que sussurramos ao vento','Amor'),
('Tu és o motivo das minhas canções mais bonitas','Amor'),
('A ternura do teu olhar é a calmaria que busco nos dias turbulentos','Amor'),

('Cada raio de sol traz consigo a promessa de dias radiantes','Felicidade'),
('O brilho das estrelas à noite é um espetáculo que encanta','Felicidade'),
('A brisa suave acaricia a alma, trazendo uma paz profunda','Felicidade'),
('A paleta de cores do pôr do sol é um presente para os olhos','Felicidade'),
('As risadas das crianças são como melodias de alegria','Felicidade'),
('A gratidão pelo presente torna o dia ainda mais especial','Felicidade'),
('Encontrar a paz interior é como descobrir um tesouro oculto','Felicidade'),
('O aroma do café pela manhã é um convite à plenitude','Felicidade'),
('A harmonia da natureza é uma sinfonia que eleva o espírito','Felicidade'),
('O calor de um abraço apertado é um refúgio de bem-estar','Felicidade'),
('Desfrutar de momentos simples é o segredo da verdadeira alegria','Felicidade'),
('A conquista de pequenas metas é um passo em direção à realização','Felicidade'),
('O silêncio da natureza é uma meditação para a alma','Felicidade'),
('A conexão com entes queridos é o verdadeiro tesouro da vida','Felicidade'),
('A descoberta de novos lugares é uma aventura emocionante','Felicidade'),
('O cheiro da chuva é uma renovação que desperta sorrisos','Felicidade'),
('As histórias guardadas nas páginas de um livro são janelas para a imaginação','Felicidade'),
('O calor do sol acaricia a pele, trazendo uma sensação de bem-estar','Felicidade'),
('O sabor de uma refeição preparada com carinho é uma festa para o paladar','Felicidade'),
('A sensação de liberdade ao dançar é uma celebração da vida','Felicidade'),
('O som suave das ondas do mar é uma melodia que acalma a mente','Felicidade'),
('A descoberta de talentos escondidos é uma jornada de autoconhecimento','Felicidade'),
('A superação de desafios é a prova de que somos mais fortes do que imaginamos','Felicidade'),
('A beleza de uma flor desabrochando é um espetáculo que inspira','Felicidade'),
('A generosidade aquece o coração, criando um ciclo virtuoso de alegria','Felicidade'),
('A sensação de propósito dá significado e plenitude à vida','Felicidade'),
('A expressão de gratidão é como um raio de sol iluminando o caminho','Felicidade'),
('A contemplação das estrelas é uma conexão com o infinito','Felicidade'),
('A brincadeira despreocupada é um elixir para a juventude interior','Felicidade'),
('A serenidade da noite traz consigo a promessa de um novo amanhecer','Felicidade'),
('O riso contagioso é uma sinfonia que ressoa em todos os corações','Felicidade'),
('A sensação de pertencimento é a base de relacionamentos enriquecedores','Felicidade'),
('A gentileza é uma corrente de felicidade que se espalha infinitamente','Felicidade'),
('O silêncio da natureza é uma prece que acalma a alma','Felicidade'),
('O abraço apertado de um amigo é um bálsamo para os momentos difíceis','Felicidade'),
('A serenidade de uma manhã tranquila é uma dádiva preciosa','Felicidade'),
('O reconhecimento do próprio crescimento traz uma satisfação interior','Felicidade'),
('A conexão com o presente é uma chave para a paz interior','Felicidade'),
('A celebração das pequenas vitórias é um banquete para a alma','Felicidade'),
('A expressão criativa é uma fonte inesgotável de alegria e realização','Felicidade'),
('O olhar de compreensão entre amigos é uma linguagem silenciosa','Felicidade'),
('A descoberta de beleza nas coisas simples é uma revelação encantadora','Felicidade'),
('A sincronia com a natureza é uma dança harmoniosa com a vida','Felicidade'),
('A lembrança de momentos felizes é um tesouro guardado no coração','Felicidade'),
('A paz que vem com a aceitação é um presente para a mente','Felicidade'),
('O cheiro da terra molhada após a chuva é uma fragrância de renovação','Felicidade'),
('A expressão autêntica de sentimentos é uma ponte para conexões mais profundas','Felicidade')]


baseteste =[
('A luz do sol acaricia minha pele, trazendo uma sensação calorosa','Alegria'),
('Rir até as lágrimas rolarem é uma terapia revigorante','Alegria'),
('A suavidade da brisa da manhã é um convite à serenidade','Alegria'),
('O sabor de uma refeição compartilhada é um banquete de emoções','Alegria'),
('O som das ondas quebrando na praia é uma melodia relaxante','Alegria'),
('A descoberta de um cantinho tranquilo para ler é um refúgio especial','Alegria'),
('A diversão de dançar sob as estrelas é uma celebração da vida','Alegria'),
('O aroma fresco da chuva desperta os sentidos e traz uma renovação','Alegria'),
('A suavidade de um abraço sincero é um bálsamo para a alma','Alegria'),
('A beleza de uma flor desabrochando revela a magia da natureza','Alegria'),
('A descontração de um dia sem planos é uma dádiva imprevista','Alegria'),
('O brilho nos olhos de quem ama é uma luz que aquece','Alegria'),
('A contemplação das estrelas é uma conexão com o vasto universo','Alegria'),
('A simplicidade de um piquenique ao ar livre é uma festa de sabores','Alegria'),
('O som de risadas espontâneas cria uma sinfonia de felicidade','Alegria'),
('A calma proporcionada pela natureza é uma terapia para a mente','Alegria'),
('A descoberta de uma nova paixão é uma jornada inspiradora','Alegria'),
('A harmonia de um dia bem aproveitado é uma nota alegre na vida','Alegria'),
('A troca de olhares cúmplices é uma linguagem silenciosa de alegria','Alegria'),
('A conquista de metas pessoais é um desfile de realizações','Alegria'),

('Teus olhos contam histórias que só meu coração compreende','Amor'),
('A harmonia dos nossos suspiros é uma canção eterna','Amor'),
('Cada toque teu é uma poesia que escrevemos juntos','Amor'),
('Nossos sorrisos formam um elo que transcende palavras','Amor'),
('Tuas mãos são o refúgio onde minha alma encontra paz','Amor'),
('Os silêncios compartilhados falam mais que mil declarações','Amor'),
('A cumplicidade dos nossos olhares é um pacto eterno','Amor'),
('Teu abraço é o lar onde minha alma se aninha feliz','Amor'),
('Nossos beijos contam segredos que só estrelas compreendem','Amor'),
('Tua presença é a melodia que embala os sonhos do meu coração','Amor'),
('Cada gesto teu é uma promessa sussurrada pelo vento','Amor'),
('Os risos compartilhados são tesouros que guardamos juntos','Amor'),
('A sintonia dos nossos passos é uma dança que nunca termina','Amor'),
('Tuas palavras são pétalas que adornam meu jardim secreto','Amor'),
('O conforto dos teus abraços é um abrigo contra tempestades','Amor'),
('Nossos suspiros formam constelações no céu do nosso afeto','Amor'),
('A jornada ao teu lado é um capítulo infindável de carinho','Amor'),
('O silêncio entre nós é uma sinfonia que ressoa eternamente','Amor'),
('A troca de olhares revela promessas feitas pela eternidade','Amor'),
('Teus beijos são notas que compõem a canção da nossa história','Amor'),

('Cada amanhecer é uma pintura de alegria no céu','Felicidade'),
('O calor do sol acaricia meu rosto, despertando sorrisos','Felicidade'),
('A brisa suave sussurra segredos de contentamento','Felicidade'),
('Os abraços apertados são portais para um mundo feliz','Felicidade'),
('Rir até as bochechas doerem é um bálsamo para a alma','Felicidade'),
('O aroma de café fresco pela manhã é uma carícia para os sentidos','Felicidade'),
('A simplicidade de um piquenique ao ar livre é um banquete de alegria','Felicidade'),
('A ternura de um olhar revela paisagens de serenidade','Felicidade'),
('A descoberta de um cantinho tranquilo é uma chave para a paz','Felicidade'),
('A gratidão transforma cada momento em uma pérola preciosa','Felicidade'),
('A música suave é uma trilha sonora que embala a vida','Felicidade'),
('Os sorrisos compartilhados são luzes cintilantes na escuridão','Felicidade'),
('A conquista de pequenas vitórias é um passo rumo à alegria','Felicidade'),
('O sabor de uma refeição preparada com carinho é um deleite do coração','Felicidade'),
('A harmonia da natureza é uma melodia que acalma a alma','Felicidade'),
('A amizade verdadeira é um tesouro radiante de bem-estar','Felicidade'),
('A serenidade de uma noite estrelada é uma poesia silenciosa','Felicidade'),
('O calor do abraço de quem amamos é um raio de felicidade','Felicidade'),
('A beleza de um pôr do sol é uma dádiva que aquece a alma','Felicidade'),
('A chuva suave é um beijo do universo, regando a alegria','Felicidade')]


#Cabeçalho
st.subheader('Aluno: Matthews Borba')

#Nome_do_usuario
user_input = st.sidebar.text_input('Digite seu nome')

st.write('Usuario: ', user_input)

#dados do usuario com a funcao

nltk.download('stopwords')
nltk.download('rslp')

stopwordsnltk = nltk.corpus.stopwords.words('portuguese')
stopwordsnltk.append('vou')
stopwordsnltk.append('tão')

# Método Aplicando Stemmer nas Palavras Identificadas (Radical das Palavras)
def aplicastemmer(texto):
    stemmer = nltk.stem.RSLPStemmer()
    frasessstemming = []
    for (palavras, emocao) in texto:
        comstemming = [str(stemmer.stem(p)) for p in palavras.split() if p not in stopwordsnltk]
        frasessstemming.append((comstemming, emocao))
    return frasessstemming

frasescomstemmingtreinamento = aplicastemmer(basetreinamento)
frasescomstemmingteste = aplicastemmer(baseteste)
#print(frasescomstemming)

def buscapalavras(frases):
    todaspalavras = []
    for (palavras, emocao) in frases:
        todaspalavras.extend(palavras)
    return todaspalavras

palavras = buscapalavras(frasescomstemmingtreinamento)

def buscafrequencia(palavras):
    palavras = nltk.FreqDist(palavras)
    return palavras

frequencia = buscafrequencia(palavras)

# Método para verificar as palavras únicas
def buscapalavrasunicas(frequencia):
    freq = frequencia.keys()
    return freq

# Chamada do método buscapalavasunicas
palavrasunicas = buscapalavrasunicas(frequencia)
#palavrasunicasteste = buscapalavrasunicas(frequenciateste)
#print(palavrasunicastreinamento)

# Extrator de Palavras.
def extratorpalavras(documento):
    doc = set(documento)
    caracteristicas = {}
    for palavras in palavrasunicas:
        caracteristicas['%s' % palavras] = (palavras in doc)
    return caracteristicas

# Chamar o ExtratorPalavras
caracteristicasfrase = extratorpalavras(['am', 'nov', 'dia'])
#print(caracteristicasfrase)

print(caracteristicasfrase)

# Extração das particularidades com o nltk.classify.apply_features
basecompletatreinamento = nltk.classify.apply_features(extratorpalavras, frasescomstemmingtreinamento)
basecompletateste = nltk.classify.apply_features(extratorpalavras, frasescomstemmingteste)

#print(basecompleta[15])
classificador = nltk.NaiveBayesClassifier.train(basecompletatreinamento)

print(nltk.classify.accuracy(classificador, basecompletateste))

print(nltk.classify.accuracy(classificador, basecompletatreinamento))

erros = []
for (frase, classe) in basecompletateste:
    print(frase)
    print(classe)
    resultado = classificador.classify(frase)
    if resultado != classe:
        erros.append((classe, resultado, frase))
for (classe, resultado, frase) in erros:
    print(classe, resultado, frase)

from nltk.metrics import ConfusionMatrix
esperado = []
previsto = []
for (frase, classe) in basecompletateste:
    resultado = classificador.classify(frase)
    previsto.append(resultado)
    esperado.append(classe)

matriz = ConfusionMatrix(esperado, previsto)
print(matriz)

st.subheader('Acurácia do modelo')
st.write(nltk.classify.accuracy(classificador, basecompletatreinamento) * 100)

def get_user_data():
    mensagem = st.sidebar.text_input('Qual a mensagem?')

    return mensagem

teste = get_user_data()
testestemming = []
stemmer = nltk.stem.RSLPStemmer()
for (palavrastreinamento) in teste.split():
    comstem = [p for p in palavrastreinamento.split()]
    testestemming.append(str(stemmer.stem(comstem[0])))
print(testestemming)

novo = extratorpalavras(testestemming)

print(classificador.classify(novo))
distribuicao = classificador.prob_classify(novo)
for classe in distribuicao.samples():
    print("%s: %f" % (classe, distribuicao.prob(classe)))

#previsao
st.subheader('Essa mensagem indica: ')
st.write(classificador.classify(novo))







