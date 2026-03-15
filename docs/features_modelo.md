# Dicionário de Features do Modelo

Este documento detalha o *breakdown* de todas as features (variáveis explicativas) que compõem o vetor de entrada do modelo de predição de sinistro (`flSinistro`), explicando o que cada métrica tenta capturar do ponto de vista do negócio.

As features estão definidas e declaradas nos módulos `src/model_sinistro/preprocessing.py` e derivadas/enriquecidas durante o script de modelagem `src/model_sinistro/train.py`.

---

## 1. Features Numéricas Históricas

Estas métricas visam capturar a regularidade de eventos passados e a volatilidade do risco baseando-se em apólices *encerradas* num período restrito antes do início de vigência de uma nova apólice (evitando *data leakage*). As janelas temporais abrangem recortes de 90, 365, 730 e 1095 dias.

### Histórico por Município
- `nrApolicesMun<janela>d`: Contagem absoluta de apólices já encerradas na janela especificada naquele município. Mostra a representatividade da série histórica.
- `nrSinistrosMun<janela>d`: Quantidade de apólices afetadas por um evento na janela temporária no município. Reconstitui a frequência com que coisas dão errado naquele local.
- `nrTaxaSinistroMun<janela>d`: Frequência relativa (Sinistros / Total de Apólices). Indica de fato a *probabilidade empírica* do município.
- `nrIndiceSeveridadeMun<janela>d`: Razão média das indenizações pagas pelo total segurado. Mede a força/impacto financeiro de eventos nesse município (alguns locais ativam o seguro, mas para perdas muito parciais; outros acionam com perda total da safra).

### Visão Comportamental do Produtor / Assimetria de Informação
- `nrApolicesAbertas30d` e `nrApolicesAbertas90d`: Volume atípico de contratações recentes no município. *Objetivo:* capturar seleção adversa "ante-fato". Se essa contagem aumenta criticamente, pode sinalizar uma percepção coletiva de um evento iminente (ex.: seca projetada num el niño).

### Visão Expandida e Baseline de Cultura/UF
Quando municípios não possuem histórico denso, o modelo herda um viés através da ótica estadual por categoria de cultura (`tipo_cultura`).
- `nrApolicesCulturaUf<janela>d` e `nrSinistrosCulturaUf<janela>d`: Contagem total de histórico no Estado para a categoria (ex: "Graos no Paraná").
- `nrTaxaSinistroCulturaUf<janela>d` e `nrSeveridadeCulturaUf<janela>d`: Médias gerais observadas no período como teto de risco base estadual.
- `nrNivelCobMedioCulturaUf365d`: Nível médio (ex: 70%, 80%) de cobertura escolhida no Estado. Subidas neste número acusam produtor assumindo menos risco individualmente e transferindo mais à seguradora.
- `nrApolicesCulturaExata365d`: A diferenciação refinada do tipo exato (ex. "Soja" em vez do aglomerado "Graos").
- `nrTaxaMediaCulturaUf365d` e `nrStdTaxaCulturaUf365d`: A métrica de inflação das seguradoras. Estados/culturas com desvio-padrão altíssimo (`nrStdTaxa`) revelam um mercado incerto em que empresas falham em estabilizar preços.

### Comportamento Histórico da Seguradora
- `nrConcentracaoSeguradora365d`: Market-share (participação de mercado) que a seguradora possui frente à cultura na região. Alta concentração traz expertise, mas eleva suscetibilidade ao agravar climático.
- `nrApolicesSegCultura365d`: Volume sob domínio daquela seguradora.
- `nrTaxaSinistroSegCultura365d` e `nrSeveridadeSegCultura365d`: A "taxa de dor" que *esta* seguradora enfrentou na cultura no último ano, diferente da média do Estado. Revela as fragilidades do seu portfólio de risco próprio.
- `nrPctCarteiraSegMun365d`: Porcentagem do patrimônio de risco de uma seguradora estacionado nesta cidade específica nos últimos 365 dias. 
- `nrHHI_seguradora_mun`: Índice de Herfindahl-Hirschman de diversificação geográfica. Indica resiliência contra eventos pontuais graves.

### Indexadores de Precificação
- `nrAnomaliaTaxa`: Indicador puramente de *pricing*. Dado como `(Taxa Apólice Atual / nrTaxaMediaCulturaUf365d)`. Funciona como radar: taxas muito baratas numa localidade perigosa ou taxas penalizadoras por histórico restrito afetam diretamente as chances daquele produtor gerar sinistro na modelagem.

---

## 2. Features Numéricas de Apólice (Engenharia Financeira)

Aqui o classificador lê o desenho do contrato, evidenciando escolhas morais de proteção e as densidades do terreno. Nada aqui contém "futuro" da apólice.

- `nrTrimestre` e `nrAnoPlantio`: Localizações cronológicas em série de safra.
- `nrDuracaoDias`: Número escalar do período coberto em dias.
- `nrDuracaoRelativa`: `(Dias / 365)`. Traz isonomia temporal para o ML quando se treina culturas de rotatividade (como hortaliças) junto de perenes (como florestas).
- `flSafraVerao`: Flag booleano (`1` se setembro-janeiro, senão `0`). Sazonalidade típica dos maiores choques climáticos no agronegócio.
- `nrDensidadeValorSegHa`: Exposição patrimonial intensa (`Valor Segurado / Área`). Regiões/Culturas mais densas costumam apresentar severidade monstruosa caso algo acorra (altos yields vs pouca terra).
- `nrPremioPorHa`: Custo base cobrado por hectare.
- `nrRazaoCoberturaProd`: Razão `(Produtividade Segurada / Produtividade Estimada)`. O *Risco Moral* da plantação. Produtores seguros de suas práticas rurais podem fazer seguro de menor fatia, apostando alto na própria tecnologia para cobrir a cota de risco nativa da produção. 
- `nrRazaoSubvencaoPremio`: Qual parcela do bolso veio amparada politicamente.
- `nrTaxaApolice`: A precificação contínua imposta pela operadora do seguro (% sobre valor).
- `nrNivelCobertura`: Métrica informada explícita de cobertura (% paramétrico).
- `nrAreaPorAnimal`: Densidade no segmento de pecuário, refletindo extensão de pasto por contingente vivo.

---

## 3. Features Categóricas

Estas contêm baixa cardinalidade via processamento String StringIndexed / One-Hot, ditando as regras chaves macroscópicas. 

- `tipo_cultura`: Agrupador botânico consolidado para diminuir a escassez dos labels (ex: *Sorgo, Milho, Aveia e Soja* entram como `graos`).
- `regiao`: Discretização geopolítica do UF (`SUL`, `SUDESTE`, `NORDESTE`, `CENTRO-OESTE`, `NORTE`). Agrupa similaridade de biomas e perfis fundiários.
- `seguradora`: Operadora da apólice, cujas práticas variam de conservadoras a predatórias.
- `nrEventosDominante365d`: A causa *mais frequente* (Moda do Histórico de `evento` != nenhum) declarada naquele binômio município/cultura do passado.

---

## 4. Features Cíclicas (Engenharia de Séries Temporais)

Tratamento especial para o mês de assinatura atenuando o efeito de rompimento entre `Mês 12` e `Mês 1`. Se usássemos mês direto como numérico, dezembro estaria a 11 números de distância de Janeiro, o que meteorologicamente é uma mentira analítica.

- `nrSinMes`: Transformada Sine aplicada sobre o Mês de Plantio (`nrMesPlantio`).
- `nrCosMes`: Transformada Cosine aplicada sobre o Mês de Plantio (`nrMesPlantio`).

> **Nota Adicional de Infraestrutura:** 
> Termos como `mun` (Código IBGE), `uf`, `cultura` (nome extenso), `apolice` e `dtRef` são **chaves exclusivas de lookup** operadas internamente na mescla com o Feature Store, e são intencionalmente podadas da matriz que faz fit/predict contornando dimensionalidade e viés ID.
