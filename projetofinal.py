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
('No brilho dos teus olhos encontro o meu lar.','Amor'),
('Cada sorriso teu é uma canção que embala meu coração.','Amor'),
('Amor é a linguagem que transcende todas as palavras.','Amor'),
('Te amar é como dançar sob as estrelas, leve e infinito.','Amor'),
('No toque das tuas mãos, encontro a paz que buscava.','Amor'),
('És a poesia que habita os versos do meu ser.','Amor'),
('O amor é a luz que ilumina os dias mais sombrios.','Amor'),
('Teu abraço é o refúgio que acalma todas as tempestades.','Amor'),
('Cada momento contigo é uma página de felicidade.','Amor'),
('Nos teus beijos, descubro o sabor do paraíso.','Amor'),
('Amor verdadeiro é como uma rosa que desabrocha a cada dia.','Amor'),
('Contigo, o tempo parece dançar em câmera lenta.','Amor'),
('A saudade é a música que o coração toca na ausência do teu ser.','Amor'),
('Te amar é encontrar sentido em cada detalhe insignificante.','Amor'),
('O amor é a força que cura todas as feridas.','Amor'),
('Tu és a razão pela qual o sol brilha mesmo nos dias nublados.','Amor'),
('Nossas almas dançam juntas na sinfonia do amor.','Amor'),
('Em teus braços, encontro o abrigo perfeito para a minha alma.','Amor'),
('O amor é a melodia que embala os sonhos mais doces.','Amor'),
('Nos teus olhos, descubro um universo de ternura.','Amor'),
('Amor é a arte de transformar o comum em extraordinário.','Amor'),
('Tu és a inspiração que colore os dias cinzentos.','Amor'),
('Amar é como plantar flores no jardim da vida.','Amor'),
('Teu amor é a bússola que guia meu coração.','Amor'),
('A cada batida do meu coração, sussurra teu nome.','Amor'),
('O amor é a magia que transforma simples momentos em memórias eternas.','Amor'),
('Tua presença é a luz que dissipa as sombras da solidão.','Amor'),
('Nossos sorrisos se entrelaçam, criando um quadro de felicidade.','Amor'),
('Amar é descobrir a beleza nas pequenas coisas da vida.','Amor'),
('Teu amor é o alicerce que sustenta a construção da nossa história.','Amor'),
('Cada desafio se torna mais fácil com teu amor ao meu lado.','Amor'),
('Amor é a chave que abre as portas da compreensão e da paciência.','Amor'),
('Nossa história é um livro escrito com as tintas vibrantes do amor.','Amor'),
('Em teus braços, encontro o conforto que o mundo não pode oferecer.','Amor'),
('Amar é como navegar em mares desconhecidos, confiando na bússola do coração.','Amor'),
('Te amar é sentir o pulsar da vida em cada batida do coração.','Amor'),
('O amor é a cola que une os fragmentos do nosso eu.','Amor'),
('Contigo, cada dia é uma página em branco pronta para ser preenchida com amor.','Amor'),
('Nosso amor é uma dança, com passos de cumplicidade e harmonia.','Amor'),
('Teu amor é o farol que guia meu barco nas águas da incerteza.','Amor'),
('Amar é encontrar na imperfeição a beleza da singularidade.','Amor'),
('Nosso amor é uma obra de arte, pintada com pinceladas de carinho e respeito.','Amor'),
('Teu amor é a melodia que embala os sonhos que dançam na noite.','Amor'),
('Amor é a força que transforma o ordinário em extraordinário.','Amor'),
('A cada palavra de amor, construímos pontes que unem nossas almas.','Amor'),
('Nosso amor é como um jardim, florescendo com a rega do cuidado.','Amor'),
('Te amar é decifrar os enigmas do coração com a chave da ternura.','Amor'),
('Amor é a resposta para as perguntas que nem mesmo sabíamos que tínhamos.','Amor'),
('Cada promessa de amor é um elo que fortalece o vínculo entre nós.','Amor'),
('Em teus abraços, encontro o porto seguro para a navegação da vida.','Amor'),
('Amar é como tecer uma colcha de retalhos, unindo momentos diversos em harmonia.','Amor'),
('Nosso amor é como uma sinfonia, onde cada nota ressoa com a harmonia da paixão.','Amor'),
('Teu amor é a inspiração que dá vida às páginas do meu livro.','Amor'),
('Amor é o fogo que aquece as noites frias da solidão.','Amor'),
('Cada beijo teu é um capítulo inesquecível na história da minha vida.','Amor'),
('Te amar é sentir a essência da felicidade pulsando em meu peito.','Amor'),
('Amor é o vínculo que une nossas almas, transcendendo tempo e espaço.','Amor'),
('Nossa jornada de amor é um conto épico, escrito nas estrelas do destino.','Amor'),
('Em teus olhos, encontro a magia que colore os dias mais cinzentos.','Amor'),
('Amor é a força que nos impulsiona a superar qualquer obstáculo.','Amor'),
('Teu sorriso é a luz que ilumina até os dias mais sombrios.','Amor'),
('Nosso amor é como um jardim secreto, florescendo com a beleza do carinho.','Amor'),
('Amar é criar memórias que resistem ao teste do tempo.','Amor'),
('Te amar é como contemplar a aurora de um novo dia, cheio de possibilidades.','Amor'),
('Amor é a música que embala a dança eterna das nossas almas.','Amor'),
('Em cada gesto de carinho, escrevemos poemas invisíveis de amor.','Amor'),
('Teu amor é a sinfonia que ecoa suavemente no silêncio da noite.','Amor'),
('Amar é como navegar pelos mares desconhecidos do coração, sem medo de se perder.','Amor'),
('Nosso amor é uma obra-prima, esculpida com os cinzéis da paciência e da compreensão.','Amor'),
('Te amar é como folhear um livro encantado, cheio de surpresas e emoções.','Amor'),
('Amor é a força que une nossos destinos, entrelaçando nossas histórias.','Amor'),
('Cada suspiro teu é uma poesia que acalma as tormentas do meu ser.','Amor'),
('Nosso amor é uma dança eterna, onde os passos são guiados pela sintonia do coração.','Amor'),
('Amar é construir castelos de felicidade no terreno sólido do companheirismo.','Amor'),
('Teu amor é o farol que guia meu barco nas águas tumultuadas da vida.','Amor'),
('Amor é a chave que abre as portas para os jardins secretos da alma.','Amor'),
('Cada olhar teu é uma declaração silenciosa do amor que habita em nós.','Amor'),
('Nosso amor é uma jornada fascinante, com capítulos cheios de aventuras e descobertas.','Amor'),
('Te amar é como escrever sonetos de paixão nas páginas do destino.','Amor'),
('Amor é a cola que une os pedaços quebrados, criando um mosaico de felicidade.','Amor'),
('Nosso amor é uma canção eterna, cuja melodia ressoa nas batidas dos nossos corações.','Amor'),
('Teu abraço é o abrigo perfeito, onde encontro paz no meio da tempestade.','Amor'),
('Amar é como voar sem medo, impulsionado pela asa do afeto.','Amor'),
('Teu amor é a tinta que colore os dias cinzentos da minha existência.','Amor'),
('Amor é a chama que arde constantemente, aquecendo os recantos mais frios do coração.','Amor'),
('Cada palavra tua é uma declaração de amor que ressoa nos recantos da minha alma.','Amor'),
('Nosso amor é como um livro aberto, pronto para ser escrito com os capítulos da felicidade.','Amor'),
('Te amar é como desvendar os mistérios do universo, uma jornada infinita e fascinante.','Amor'),
('Amor é a luz que guia nossos passos nas estradas sinuosas da vida.','Amor'),
('Nosso amor é uma constelação brilhante, pontilhada pelas estrelas da cumplicidade.','Amor'),
('Teu amor é o alicerce que sustenta a construção dos sonhos que compartilhamos.','Amor'),
('Amar é como tecer uma tapeçaria de afeto, entrelaçando fios de carinho e ternura.','Amor'),
('Nosso amor é a trilha sonora que embala os momentos mais significativos das nossas vidas.','Amor'),
('Te amar é como folhear as páginas de um livro encantado, repleto de emoções intensas.','Amor'),
('Amor é a sinfonia que ressoa suavemente, criando harmonia nos corações apaixonados.','Amor'),
('Cada beijo teu é um verso apaixonado, escrito no poema eterno do nosso amor.','Amor'),
('Teu sorriso é a luz que ilumina até os dias mais sombrios, trazendo calor ao meu ser.','Amor'),
('Amor é o fio invisível que conecta nossas almas, criando um laço indestrutível.','Amor'),
('Nosso amor é uma dança eterna, onde os passos são guiados pela música do coração.','Amor'),
('Te amar é como viver um conto de fadas, onde a felicidade é o final feliz que buscamos.','Amor'),

('O sol radiante pela manhã é um convite à alegria','Alegria'),
('A risada de uma criança é a sinfonia mais pura da alegria','Alegria'),
('Encontrar um velho amigo é uma explosão instantânea de alegria','Alegria'),
('Cada pôr do sol pinta o céu com cores da alegria','Alegria'),
('A alegria está nas pequenas vitórias diárias da vida','Alegria'),
('A música alegre é a trilha sonora da alma contente','Alegria'),
('Um abraço apertado pode derreter até o mais frio dos corações, trazendo alegria','Alegria'),
('A descoberta de algo novo é uma fonte inesgotável de alegria','Alegria'),
('A alegria contagia como um sorriso sincero','Alegria'),
('A gratidão transforma qualquer situação em uma fonte de alegria','Alegria'),
('Cada amanhecer é uma promessa renovada de alegria','Alegria'),
('A dança é a expressão física da alegria que transborda','Alegria'),
('A alegria está nas experiências compartilhadas com quem amamos','Alegria'),
('A gentileza é um gesto que floresce no jardim da alegria','Alegria'),
('A alegria está nos detalhes que muitas vezes passam despercebidos','Alegria'),
('A leitura de um livro cativante é uma jornada na qual a alegria se manifesta','Alegria'),
('A conquista de um objetivo é uma explosão de alegria merecida','Alegria'),
('A natureza, com sua beleza infinita, é uma fonte constante de alegria','Alegria'),
('A alegria está em dar sem esperar nada em troca','Alegria'),
('A companhia de animais de estimação é uma fonte pura de alegria','Alegria'),
('A alegria reside na aceitação plena de quem somos','Alegria'),
('A arte de criar é uma jornada repleta de alegria e autodescoberta','Alegria'),
('A alegria está em apreciar o momento presente','Alegria'),
('A liberdade de ser autêntico é um caminho para a alegria duradoura','Alegria'),
('A alegria é um raio de luz que ilumina os dias mais sombrios','Alegria'),
('A expressão criativa é uma explosão de alegria que vem de dentro','Alegria'),
('A alegria está em aprender algo novo a cada dia','Alegria'),
('A generosidade é um ato que semeia campos de alegria','Alegria'),
('A alegria está em encontrar beleza nas coisas simples','Alegria'),
('A arte de rir de si mesmo é uma fonte inesgotável de alegria','Alegria'),
('A alegria está na jornada, não apenas no destino','Alegria'),
('A música alegre ressoa como um bálsamo para a alma','Alegria'),
('A alegria é contagiante, espalhando-se como ripples na água','Alegria'),
('A alegria está em abraçar a diversidade que a vida oferece','Alegria'),
('A celebração das conquistas alheias traz uma onda de alegria','Alegria'),
('A alegria está em encontrar soluções criativas para desafios','Alegria'),
('A conexão com a natureza é uma fonte serena de alegria','Alegria'),
('A alegria está em compartilhar sorrisos com estranhos','Alegria'),
('A simplicidade é o caminho para a alegria descomplicada','Alegria'),
('A gratidão é a chave que abre a porta para a alegria duradoura','Alegria'),
('A alegria está em ser autêntico, sem máscaras ou disfarces','Alegria'),
('A dança da chuva é uma celebração natural que desperta alegria','Alegria'),
('A alegria está em cultivar relacionamentos sinceros','Alegria'),
('A empatia é uma ponte que conecta corações, trazendo alegria','Alegria'),
('A alegria está em aceitar e amar as imperfeições','Alegria'),
('A contemplação de um céu estrelado é um convite à alegria silenciosa','Alegria'),
('A autenticidade é o solo fértil onde floresce a alegria','Alegria'),
('A alegria está em apreciar o espetáculo da vida cotidiana','Alegria'),
('A generosidade de espírito é uma fonte infinita de alegria','Alegria'),
('A alegria está em ser grato pelo que se tem','Alegria'),
('A risada é a linguagem universal da alegria','Alegria'),
('A alegria está em encontrar beleza nas diferenças','Alegria'),
('A harmonia da música é uma expressão da alegria em forma sonora','Alegria'),
('A alegria está em viver de acordo com os valores mais verdadeiros','Alegria'),
('A descoberta de um talento é uma fonte inesgotável de alegria','Alegria'),
('A alegria está em abraçar o desconhecido com entusiasmo','Alegria'),
('A contemplação de uma obra de arte é uma viagem à alegria estética','Alegria'),
('A simplicidade de um momento quieto traz uma alegria serena','Alegria'),
('A alegria está em encontrar propósito em cada ação','Alegria'),
('A celebração de conquistas, grandes ou pequenas, é uma festa de alegria','Alegria'),
('A alegria está em deixar ir o que não pode ser mudado','Alegria'),
('A gratidão é um eco de alegria que ressoa no coração','Alegria'),
('A alegria está em abraçar a mudança como uma oportunidade','Alegria'),
('A companhia de amigos verdadeiros é um tesouro de alegria','Alegria'),
('A alegria está em cultivar um coração cheio de compaixão','Alegria'),
('A luz suave da manhã traz consigo uma sensação acolhedora de alegria','Alegria'),
('A alegria está em saborear os momentos simples com apreciação','Alegria'),
('A aceitação incondicional traz consigo uma onda de alegria interior','Alegria'),
('A alegria está em dançar sob a chuva, celebrando a vida','Alegria'),
('A maravilha da descoberta é uma fonte de alegria renovada','Alegria'),
('A alegria está em ser a razão do sorriso de alguém','Alegria'),
('A alegria reside na jornada, não no destino final','Alegria'),
('A liberdade de expressão é um caminho para a alegria autêntica','Alegria'),
('A alegria está em ser fiel a si mesmo, sem comprometer a autenticidade','Alegria'),
('A expressão artística é uma explosão de alegria que flui da alma','Alegria'),
('A alegria está em encontrar inspiração nas coisas simples da vida','Alegria'),
('A alegria está em ser um raio de sol na vida de alguém','Alegria'),
('A celebração das diferenças é uma manifestação de alegria coletiva','Alegria'),
('A alegria está em permitir-se ser vulnerável e autêntico','Alegria'),
('A prática da gratidão é um portal para a alegria duradoura','Alegria'),
('A alegria está em abraçar a jornada com coragem e otimismo','Alegria'),
('A música alegre ressoa como um chamado para dançar com a vida','Alegria'),
('A alegria está em celebrar as pequenas vitórias diárias','Alegria'),
('A conexão com a natureza é uma fonte infinita de alegria','Alegria'),
('A alegria está em sorrir, mesmo nos momentos mais desafiadores','Alegria'),
('A expressão criativa é um rio de alegria que flui ininterruptamente','Alegria'),
('A alegria está em encontrar beleza nas diferentes estações da vida','Alegria'),
('A gratidão é uma canção suave que embala a alma com alegria','Alegria'),
('A alegria está em apreciar o presente sem ser sobrecarregado pelo passado ou futuro','Alegria'),
('A simplicidade é a chave para a alegria descomplicada','Alegria'),
('A celebração das conquistas alheias é uma festa de alegria compartilhada','Alegria'),
('A alegria está em permitir-se ser tocado pela beleza ao seu redor','Alegria'),
('A autenticidade é um farol que guia para a alegria interior','Alegria'),
('A alegria está em encontrar sentido e propósito em cada ação','Alegria'),
('A risada é um elixir de alegria que cura os corações cansados','Alegria'),
('A alegria está em criar memórias duradouras com aqueles que amamos','Alegria'),
('A generosidade é um ato que semeia campos de alegria coletiva','Alegria'),
('A alegria está em abraçar a jornada com entusiasmo e curiosidade','Alegria'),
('A contemplação silenciosa é uma porta aberta para a alegria interior','Alegria'),
('A alegria está em cultivar uma mente grata e um coração amoroso','Alegria'),

('A felicidade está nas pequenas alegrias do cotidiano','Felicidade'),
('Cada amanhecer traz consigo a promessa de felicidade renovada','Felicidade'),
('A conexão genuína com os outros é o caminho para a felicidade','Felicidade'),
('A descoberta de paixões é uma jornada rumo à felicidade plena','Felicidade'),
('A felicidade está em abraçar a gratidão pelo que temos','Felicidade'),
('A risada espontânea é a trilha sonora da verdadeira felicidade','Felicidade'),
('A aceitação de si mesmo é o alicerce da felicidade duradoura','Felicidade'),
('A felicidade está em viver autenticamente, sem máscaras','Felicidade'),
('A busca pelo equilíbrio interior é um caminho para a felicidade','Felicidade'),
('A felicidade reside na aceitação plena do momento presente','Felicidade'),
('O amor-próprio é o primeiro passo em direção à felicidade plena','Felicidade'),
('A expressão criativa é uma fonte rica de alegria e felicidade','Felicidade'),
('A conquista de metas pessoais é uma fonte inesgotável de felicidade','Felicidade'),
('A felicidade está em celebrar as conquistas, por menores que sejam','Felicidade'),
('A jornada para a autodescoberta é pavimentada com momentos de felicidade','Felicidade'),
('A conexão com a natureza é um portal para a felicidade serena','Felicidade'),
('A felicidade está em compartilhar alegrias com aqueles que amamos','Felicidade'),
('A gratidão é o solo fértil onde floresce a árvore da felicidade','Felicidade'),
('A felicidade está em encontrar beleza na simplicidade da vida','Felicidade'),
('A celebração das diferenças é um tributo à riqueza da felicidade','Felicidade'),
('A generosidade é um ato que multiplica a felicidade, tanto para quem dá quanto para quem recebe','Felicidade'),
('A felicidade está em viver com propósito e significado','Felicidade'),
('A autoexpressão autêntica é um passo em direção à felicidade verdadeira','Felicidade'),
('A empatia é um elo que conecta corações, gerando ondas de felicidade','Felicidade'),
('A felicidade está em aprender com as adversidades e crescer com elas','Felicidade'),
('A música alegre é uma linguagem universal que transcende barreiras, trazendo felicidade','Felicidade'),
('A prática da gentileza é um atalho para a estrada da felicidade','Felicidade'),
('A felicidade está em sintonizar a mente com pensamentos positivos','Felicidade'),
('A aceitação das imperfeições é uma chave para a felicidade plena','Felicidade'),
('A felicidade reside na jornada, não apenas no destino final','Felicidade'),
('A simplicidade é o segredo da felicidade descomplicada','Felicidade'),
('A descoberta de novas experiências é uma fonte inesgotável de felicidade','Felicidade'),
('A felicidade está em cultivar relacionamentos saudáveis e significativos','Felicidade'),
('A autenticidade é um farol que guia para o porto seguro da felicidade interior','Felicidade'),
('A gratidão é uma poção mágica que amplifica a felicidade','Felicidade'),
('A felicidade está em abraçar a mudança como uma oportunidade de crescimento','Felicidade'),
('A jornada espiritual é um caminho para a verdadeira felicidade','Felicidade'),
('A felicidade está em viver de acordo com os valores mais autênticos','Felicidade'),
('A contemplação silenciosa é uma porta aberta para a felicidade interior','Felicidade'),
('A celebração das pequenas vitórias diárias é uma festa de felicidade cotidiana','Felicidade'),
('A liberdade de ser autêntico é um caminho para a felicidade sem reservas','Felicidade'),
('A alegria compartilhada é a multiplicação da felicidade','Felicidade'),
('A felicidade está em abraçar a jornada com coragem e otimismo','Felicidade'),
('A prática da compaixão é um atalho para a felicidade duradoura','Felicidade'),
('A autoaceitação é o segredo para desbloquear a porta da felicidade interior','Felicidade'),
('A felicidade está em cultivar um coração grato','Felicidade'),
('A busca por significado é uma estrada pavimentada com momentos de felicidade','Felicidade'),
('A felicidade está em viver no presente, aproveitando cada momento','Felicidade'),
('A beleza da vida está na diversidade que ela oferece, gerando riqueza de felicidade','Felicidade'),
('A felicidade reside na celebração das jornadas individuais e coletivas','Felicidade'),
('A descoberta de um propósito pessoal é uma fonte duradoura de felicidade','Felicidade'),
('A felicidade está em cultivar um coração aberto e amoroso','Felicidade'),
('A celebração da própria jornada é um tributo à felicidade conquistada','Felicidade'),
('A prática da bondade é um portal para a felicidade compartilhada','Felicidade'),
('A felicidade está em abraçar a vulnerabilidade como uma expressão autêntica','Felicidade'),
('A gratidão é o farol que ilumina o caminho da felicidade duradoura','Felicidade'),
('A alegria compartilhada é um elo que fortalece os laços de felicidade','Felicidade'),
('A felicidade está em viver em harmonia consigo mesmo e com o mundo','Felicidade'),
('A contemplação silenciosa é uma prática que nutre a felicidade interior','Felicidade'),
('A expressão criativa é uma explosão de felicidade que flui da alma','Felicidade'),
('A autenticidade é uma estrada para a felicidade sem destino','Felicidade'),
('A celebração das conquistas alheias é uma festa de felicidade compartilhada','Felicidade'),
('A felicidade está em viver com propósito e significado','Felicidade'),
('A gratidão é um presente que amplifica a felicidade recebida','Felicidade'),
('A jornada espiritual é uma trilha para a felicidade eterna','Felicidade'),
('A felicidade está em viver em alinhamento com os valores mais profundos','Felicidade'),
('A celebração das pequenas vitórias diárias é um banquete de felicidade','Felicidade'),
('A autenticidade é um farol que ilumina o caminho da felicidade verdadeira','Felicidade'),
('A felicidade está em abraçar a jornada com coragem e aceitação','Felicidade'),
('A expressão criativa é um oceano de felicidade que flui sem limites','Felicidade'),
('A gratidão é uma fonte inesgotável de felicidade duradoura','Felicidade'),
('A contemplação silenciosa é um portal para a felicidade interior','Felicidade'),
('A felicidade está em aceitar a complexidade da vida e encontrar beleza nela','Felicidade'),
('A celebração das conquistas alheias é um ato de felicidade compartilhada','Felicidade'),
('A autoaceitação é uma chave que destranca a porta para a felicidade duradoura','Felicidade'),
('A alegria compartilhada é uma dança de felicidade coletiva','Felicidade'),
('A felicidade está em viver em equilíbrio com o corpo, mente e espírito','Felicidade'),
('A expressão criativa é uma explosão de felicidade que colore a vida','Felicidade'),
('A gratidão é um eco de felicidade que ressoa infinitamente','Felicidade'),
('A celebração das pequenas alegrias é um festival de felicidade cotidiana','Felicidade'),
('A autenticidade é um farol que guia para a felicidade sem restrições','Felicidade'),
('A felicidade está em viver no presente, saboreando cada momento','Felicidade'),
('A contemplação silenciosa é uma fonte de felicidade interior inesgotável','Felicidade'),
('A expressão criativa é uma dança de felicidade que flui da alma','Felicidade'),
('A gratidão é um perfume que exala a fragrância da felicidade duradoura','Felicidade'),
('A celebração das conquistas alheias é uma expressão de felicidade coletiva','Felicidade'),
('A autenticidade é uma estrada pavimentada com tijolos dourados de felicidade','Felicidade'),
('A felicidade está em cultivar relacionamentos autênticos e significativos','Felicidade'),
('A expressão criativa é um poema de felicidade escrito com as tintas da alma','Felicidade'),
('A gratidão é uma chama que aquece o coração, irradiando felicidade','Felicidade'),
('A celebração das pequenas vitórias diárias é uma festa de felicidade cotidiana','Felicidade'),
('A autenticidade é um farol que ilumina o caminho para a felicidade interior','Felicidade'),
('A felicidade está em viver em harmonia com a natureza e consigo mesmo','Felicidade'),
('A contemplação silenciosa é uma meditação que nutre a felicidade interior','Felicidade'),
('A expressão criativa é um rio de felicidade que flui ininterruptamente','Felicidade'),
('A gratidão é uma sinfonia que ressoa com os acordes suaves da felicidade','Felicidade'),
('A celebração das conquistas alheias é uma dança de felicidade compartilhada','Felicidade'),
('A autenticidade é um farol que guia para a felicidade sem fronteiras','Felicidade'),
('A felicidade está em viver com um coração grato e uma mente serena','Felicidade'),
('A expressão criativa é uma explosão de felicidade que brota do mais profundo ser','Felicidade')]

baseteste =[
    ('Nosso amor é uma dança eterna, entrelaçando os passos da vida.', 'Amor'),
    ('Te amar é como folhear as páginas de um livro mágico, cheio de encantos.', 'Amor'),
    ('O toque suave dos teus lábios é a melodia que embala meu coração.', 'Amor'),
    ('Cada olhar trocado entre nós conta uma história de amor silenciosa.', 'Amor'),
    ('Amor é a luz que ilumina os cantos mais escuros da alma.', 'Amor'),
    ('Teu abraço é o refúgio seguro onde encontro paz e calor.', 'Amor'),
    ('Amar é como plantar flores no jardim do coração, cultivando a beleza do afeto.', 'Amor'),
    ('Nosso amor é uma sinfonia de emoções, tocada pelos acordes da paixão.', 'Amor'),
    ('Teu sorriso é o sol que dissipa as nuvens da tristeza em meu dia.', 'Amor'),
    ('Amor é o laço invisível que une nossas almas, resistindo ao teste do tempo.', 'Amor'),
    ('Te amar é descobrir constantemente novos horizontes de felicidade.', 'Amor'),
    ('Cada palavra tua é uma declaração de amor que ressoa no eco do coração.', 'Amor'),
    ('Amar é como tecer uma colcha de memórias, entrelaçando momentos inesquecíveis.', 'Amor'),
    ('Nosso amor é uma jornada, e cada desafio fortalece os laços que nos unem.', 'Amor'),
    ('Teu toque é a carícia que acalma a tempestade que habita em mim.', 'Amor'),
    ('Amor é a essência que permeia nossos dias, transformando-os em poesia.', 'Amor'),
    ('Te amar é como explorar um universo de sentimentos, vasto e infinito.', 'Amor'),
    ('Cada beijo trocado é um capítulo ardente no livro da nossa história.', 'Amor'),
    ('Nosso amor é como um quadro abstrato, cheio de cores vibrantes e formas únicas.', 'Amor'),
    ('Amar é como dançar sob as estrelas, guiados pela melodia do coração.', 'Amor'),
    ('Teu olhar é o farol que guia meu barco nas águas tumultuadas da vida.', 'Amor'),
    ('Amor é a força que supera as adversidades, tornando-nos mais resilientes.', 'Amor'),
    ('Te amar é como colher as estrelas do céu para iluminar nossos dias.', 'Amor'),
    ('Cada suspiro teu é uma poesia que embala a serenidade do meu ser.', 'Amor'),
    ('Amor é a mágica que transforma momentos simples em lembranças eternas.', 'Amor'),
    ('Teu abraço é o porto seguro onde ancoro meu coração, seguro e sereno.', 'Amor'),
    ('Nosso amor é uma obra-prima, pintada com pinceladas de carinho e cumplicidade.', 'Amor'),
    ('Amar é como criar arte, moldando o barro da vida com as mãos do coração.', 'Amor'),
    ('Te amar é como desvendar os mistérios de um livro emocionante, página por página.', 'Amor'),
    ('Amor é a chama que arde eternamente, aquecendo os invernos da existência.', 'Amor'),
    ('Nosso amor é uma melodia única, composta pelos acordes da nossa sintonia.', 'Amor'),
    ('Teu sorriso é a luz que ilumina o caminho escuro, dissipando as sombras.', 'Amor'),
    ('Amor é a poesia que escrevemos juntos, versos que ecoam na eternidade.', 'Amor'),
    ('Te amar é como viajar por terras desconhecidas, explorando novos sentimentos.', 'Amor'),
    ('Amor é a tinta que colore as páginas em branco do livro da nossa história.', 'Amor'),
    ('Teu toque é a magia que transforma momentos simples em lembranças mágicas.', 'Amor'),
    ('Amar é como construir um castelo de sonhos, tijolo por tijolo, juntos.', 'Amor'),
    ('Nosso amor é como um jardim florescendo, cada estação trazendo novas cores.', 'Amor'),
    ('Te amar é como escrever uma canção, com notas de ternura e harmonia.', 'Amor'),
    ('Amor é a força que une nossos destinos, entrelaçando nossas almas para sempre.', 'Amor'),

    ('Ao ver o pôr do sol, uma onda de Alegria invade meu coração', 'Alegria'),
    ('O aroma do café pela manhã é um convite à Alegria diária', 'Alegria'),
    ('Na simplicidade de um sorriso, encontro a verdadeira Alegria', 'Alegria'),
    ('A chuva leve traz consigo a Alegria de renovar a natureza', 'Alegria'),
    ('Explorar novos lugares é uma jornada repleta de Alegria', 'Alegria'),
    ('A conquista de objetivos é um motivo genuíno de Alegria', 'Alegria'),
    ('O reencontro com amigos queridos é uma fonte de Alegria', 'Alegria'),
    ('A música alegre faz os corações dançarem de Alegria', 'Alegria'),
    ('A generosidade aquece o mundo com raios de Alegria', 'Alegria'),
    ('O abraço apertado de um ente querido transmite pura Alegria', 'Alegria'),
    ('Cada amanhecer traz a promessa de um novo dia cheio de Alegria', 'Alegria'),
    ('A gratidão transforma momentos simples em eternas lembranças de Alegria', 'Alegria'),
    ('A descoberta de paixões compartilhadas é uma fonte inesgotável de Alegria', 'Alegria'),
    ('A superação de desafios traz consigo uma Alegria indescritível', 'Alegria'),
    ('O brilho nos olhos revela a intensidade da Alegria interior', 'Alegria'),
    ('Compartilhar risadas com amigos é um bálsamo para a alma de Alegria', 'Alegria'),
    ('A beleza das flores em plena primavera desperta uma Alegria contagiante', 'Alegria'),
    ('A solidariedade cria laços que nutrem o coração de Alegria', 'Alegria'),
    ('A dança é uma expressão sublime da Alegria que pulsa dentro de nós', 'Alegria'),
    ('O sucesso dos outros nos enche de Alegria sincera', 'Alegria'),
    ('A serenidade de um momento tranquilo é uma fonte de Alegria interior', 'Alegria'),
    ('A alegria de aprender algo novo ilumina cada descoberta', 'Alegria'),
    ('O cheiro de chuva no ar traz consigo uma Alegria revigorante', 'Alegria'),
    ('O compartilhamento de histórias felizes reforça os laços de Alegria', 'Alegria'),
    ('A leve brisa do mar traz consigo uma sensação única de Alegria', 'Alegria'),
    ('A harmonia das cores em um quadro desperta uma Alegria estética', 'Alegria'),
    ('A fé renovada é um caminho de constante Alegria', 'Alegria'),
    ('A gentileza inesperada de um estranho cria uma onda de Alegria', 'Alegria'),
    ('A contemplação de um céu estrelado é um convite à Alegria cósmica', 'Alegria'),
    ('A camaradagem genuína entre colegas de trabalho traz Alegria ao ambiente', 'Alegria'),
    ('A serendipidade de encontrar algo perdido é uma alegria inesperada', 'Alegria'),
    ('A energia contagiante de um evento alegre é como um banquete de Alegria', 'Alegria'),
    ('A ternura de um gesto simples pode aquecer o coração de Alegria', 'Alegria'),
    ('A liberdade de expressão traz consigo uma Alegria autêntica', 'Alegria'),
    ('O som da risada de uma criança é a melodia da verdadeira Alegria', 'Alegria'),
    ('A prática da gratidão é um exercício diário de cultivar Alegria', 'Alegria'),
    ('O otimismo resiliente é uma fonte perene de Alegria interior', 'Alegria'),
    ('A celebração das pequenas vitórias diárias é uma expressão de Alegria', 'Alegria'),
    ('A beleza da diversidade cultural é uma fonte rica de Alegria compartilhada', 'Alegria'),
    ('A simples lembrança de momentos felizes é um tesouro de Alegria duradoura', 'Alegria'),

    ('Ao realizar um sonho, a Felicidade se torna tangível', 'Felicidade'),
    ('A conexão profunda com quem amamos é o alicerce da Felicidade', 'Felicidade'),
    ('Encontrar equilíbrio na vida é trilhar o caminho da verdadeira Felicidade', 'Felicidade'),
    ('A gratidão transforma momentos simples em pura Felicidade', 'Felicidade'),
    ('A superação de desafios é uma fonte inesgotável de Felicidade', 'Felicidade'),
    ('Cada sorriso sincero é um reflexo da Felicidade interior', 'Felicidade'),
    ('A descoberta de novas paixões é uma jornada repleta de Felicidade', 'Felicidade'),
    ('A compreensão profunda traz consigo uma onda de Felicidade', 'Felicidade'),
    ('O compartilhamento de alegrias duplica a intensidade da Felicidade', 'Felicidade'),
    ('A conquista de metas é um marco significativo na jornada da Felicidade', 'Felicidade'),
    ('A empatia genuína é um elixir para a Felicidade coletiva', 'Felicidade'),
    ('A liberdade de ser autêntico é o caminho para a verdadeira Felicidade', 'Felicidade'),
    ('A serenidade de momentos tranquilos é um bálsamo para a alma em busca de Felicidade', 'Felicidade'),
    ('A celebração das pequenas vitórias diárias é uma expressão viva de Felicidade', 'Felicidade'),
    ('A comunhão com a natureza desperta uma Felicidade que vem das raízes', 'Felicidade'),
    ('O apoio de amigos verdadeiros é um tesouro inestimável de Felicidade', 'Felicidade'),
    ('A autocompaixão é a chave para desbloquear a Felicidade interior', 'Felicidade'),
    ('O aprendizado constante traz uma Felicidade que se renova a cada descoberta', 'Felicidade'),
    ('A simplicidade do viver no presente é o segredo da Felicidade duradoura', 'Felicidade'),
    ('A harmonia de interesses compartilhados cria uma sinfonia de Felicidade', 'Felicidade'),
    ('A expressão criativa é uma porta aberta para a Felicidade sem limites', 'Felicidade'),
    ('A gentileza espontânea é como um raio de Felicidade irradiando a todos', 'Felicidade'),
    ('A confiança no processo da vida é o alicerce da Felicidade plena', 'Felicidade'),
    ('A contemplação silenciosa é um refúgio para a Felicidade introspectiva', 'Felicidade'),
    ('A honestidade consigo mesmo é o primeiro passo para a conquista da Felicidade', 'Felicidade'),
    ('A partilha de conquistas com entes queridos multiplica a Felicidade', 'Felicidade'),
    ('A viagem interior em busca de autoconhecimento é um caminho para a Felicidade duradoura', 'Felicidade'),
    ('O abraço caloroso de um amigo é um presente precioso de Felicidade', 'Felicidade'),
    ('A prática da gratidão diária é um exercício que nutre a Felicidade', 'Felicidade'),
    ('A confiança nas possibilidades futuras é uma fonte de Felicidade resiliente', 'Felicidade'),
    ('A descoberta de novas culturas é uma aventura que enriquece a Felicidade', 'Felicidade'),
    ('A busca por propósito dá significado à jornada em direção à Felicidade', 'Felicidade'),
    ('A serendipidade de momentos inesperadamente felizes é uma bênção de Felicidade', 'Felicidade'),
    ('A expressão de amor nutre o solo fértil da Felicidade compartilhada', 'Felicidade'),
    ('O perdão sincero é um portal para a Felicidade renovada', 'Felicidade'),
    ('A compaixão pelos outros é uma ponte que conduz à Felicidade mútua', 'Felicidade'),
    ('A aceitação plena de si mesmo é a chave para a porta da Felicidade interior', 'Felicidade'),
    ('A fé no caminho que se desenha à frente é uma luz que guia para a Felicidade', 'Felicidade'),
    ('A curiosidade constante é um convite à descoberta constante de Felicidade', 'Felicidade'),
    ('A amizade sincera é um tesouro eterno que ilumina o caminho da Felicidade', 'Felicidade')]


#Cabeçalho
st.subheader('Aluno: Matthews Borba Correia de Brito')

#Nome_do_usuario
user_input = st.text_input('Digite seu nome')

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

def get_user_data():
    mensagem = st.text_input('Qual a mensagem?')

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







