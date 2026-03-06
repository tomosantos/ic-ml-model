# =============================================================================
# const.py — Constantes globais do pipeline Bronze → Silver (Seguro Rural)
#
# Uso no Databricks: adicione `# MAGIC %run ../lib/const` no início do notebook.
# Todas as variáveis aqui declaradas ficam disponíveis no escopo do notebook.
# =============================================================================

# -----------------------------------------------------------------------------
# Caminhos e nomes de tabelas
# -----------------------------------------------------------------------------

# Unity Catalog — Camada Bronze
TABLE_BRONZE_HISTORICAL = '01_bronze.seg_rural.historical_seg'
TABLE_BRONZE_ATUAL      = '01_bronze.seg_rural.seg_2025'

# Unity Catalog — Camada Silver
TABLE_SILVER_CLEANED    = '02_silver.seg_rural.seg_cleaned'

# Unity Catalog — Camada Gold (features e labels separados)
TABLE_GOLD_FEATURES     = '03_gold.seg_rural.fs_seguro_features'
TABLE_GOLD_LABELS       = '03_gold.seg_rural.fs_seguro_labels'

# Feature Store — tabelas de agregações point-in-time
TABLE_FS_HISTORICO_MUN      = 'feature_store.seguro.fs_historico_municipio'
TABLE_FS_RISCO_CULTURA_UF   = 'feature_store.seguro.fs_risco_cultura_uf'
TABLE_FS_APOLICE_FINANCEIRO = 'feature_store.seguro.fs_apolice_financeiro'

# Volume com arquivo auxiliar do IBGE de distritos
# Fonte: https://geoftp.ibge.gov.br/organizacao_do_territorio/estrutura_territorial/divisao_territorial/2022/
DISTRITOS_PATH = '/Volumes/00_raw/data/seguro_rural/RELATORIO_DTB_BRASIL_DISTRITO.xls'

# CSV público com códigos de municípios (Patrícia Siqueira / IBGE)
IBGE_CSV_URL = 'https://patriciasiqueira.github.io/arquivos/codigos-mun.csv'

# -----------------------------------------------------------------------------
# Transliteração de acentos
# -----------------------------------------------------------------------------

ACENTOS     = 'áàâãéèêíìîóòôõúùûçÁÀÂÃÉÈÊÍÌÎÓÒÔÕÚÙÛÇ'
SEM_ACENTOS = 'aaaaeeeiiioooouuucAAAAEEEIIIOOOOUUUC'

# -----------------------------------------------------------------------------
# Colunas a remover (metadados / localização / identificação pessoal)
# -----------------------------------------------------------------------------

COLUNAS_RETIRAR = [
  'CD_PROCESSO_SUSEP', 'NR_PROPOSTA', 'ID_PROPOSTA',
  'DT_PROPOSTA',
  'NM_SEGURADO', 'NR_DOCUMENTO_SEGURADO',
  'NR_GRAU_LAT', 'NR_MIN_LAT', 'NR_SEG_LAT',
  'NR_GRAU_LONG', 'NR_MIN_LONG', 'NR_SEG_LONG',
  'DT_APOLICE', 'ANO_APOLICE',
]

# -----------------------------------------------------------------------------
# Renomeação de colunas (nome original SISSER → nome canônico do projeto)
# -----------------------------------------------------------------------------

RENAME_MAP = {
  'NM_RAZAO_SOCIAL':           'seguradora',
  'NM_MUNICIPIO_PROPRIEDADE':  'nome_mun',
  'SG_UF_PROPRIEDADE':         'uf',
  'NM_CLASSIF_PRODUTO':        'tipo',
  'NM_CULTURA_GLOBAL':         'cultura',
  'NR_AREA_TOTAL':             'area',
  'NR_ANIMAL':                 'animal',
  'NR_PRODUTIVIDADE_ESTIMADA': 'prod_est',
  'NR_PRODUTIVIDADE_SEGURADA': 'prod_seg',
  'NivelDeCobertura':          'nivel_cob',
  'VL_LIMITE_GARANTIA':        'total_seg',
  'VL_PREMIO_LIQUIDO':         'premio',
  'PE_TAXA':                   'taxa',
  'VL_SUBVENCAO_FEDERAL':      'subvencao',
  'NR_APOLICE':                'apolice',
  'CD_GEOCMU':                 'mun',
  'VALOR_INDENIZACAO':         'indenizacao',  # coluna já normalizada (sem acento)
  'EVENTO_PREPONDERANTE':      'evento',
  'DT_INICIO_VIGENCIA':        'dt_inicio_vigencia',
  'DT_FIM_VIGENCIA':           'dt_fim_vigencia',
  'NR_DECIMAL_LATITUDE':       'lat',
  'NR_DECIMAL_LONGITUDE':      'lon',
}

# -----------------------------------------------------------------------------
# Mapeamento de tipos de seguro
# Chaves sem acento (normalizadas pela etapa anterior ao rename)
# -----------------------------------------------------------------------------

TIPO_MAP = {
  'CUSTEIO':       'custeio',
  'PRODUTIVIDADE': 'produtividade',
  'FLORESTAS':     'florestas',
  'RECEITA':       'receita',
  'PECUARIO':      'pecuario',
}

# -----------------------------------------------------------------------------
# Mapeamento de eventos preponderantes
# Chaves sem acento (normalizadas)
# -----------------------------------------------------------------------------

EVENTO_MAP = {
  '0':                                  'nenhum',
  'SECA':                               'seca',
  'GEADA':                              'geada',
  'CHUVA EXCESSIVA':                    'chuva',
  'GRANIZO':                            'granizo',
  'VENTOS FORTES/FRIOS':                'vento',
  "INUNDACAO/TROMBA D'AGUA":            'inundacao',
  'VARIACAO EXCESSIVA DE TEMPERATURA':  'temp.',
  'INCENDIO':                           'incendio',
  'DEMAIS CAUSAS':                      'outras',
  'VARIACAO DE PRECO':                  'var. preco',
  'REPLANTIO':                          'replantio',
  'RAIO':                               'raio',
  'MORTE':                              'morte',
  'QUEDA DE PARREIRAL':                 'queda parr.',
  'PERDA DE QUALIDADE':                 'perda qual.',
  'DOENCAS E PRAGAS':                   'doencas',
}

# -----------------------------------------------------------------------------
# Categorização de culturas
# Chaves sem acento (normalizadas)
# -----------------------------------------------------------------------------

TIPO_CULTURA_MAP = {
  'Soja':              'graos',        'Milho 1a safra':    'graos',
  'Milho 2a safra':    'graos',        'Trigo':             'graos',
  'Sorgo':             'graos',        'Arroz':             'graos',
  'Aveia':             'graos',        'Canola':            'graos',
  'Algodao':           'graos',        'Triticale':         'graos',
  'Girassol':          'graos',        'Cevada':            'sementes',
  'Cana-de-acucar':    'perenes',      'Cafe':              'perenes',
  'Pecuario':          'perenes',
  'Feijao 1a safra':   'leguminosas',  'Feijao 2a safra':   'leguminosas',
  'Amendoim':          'leguminosas',  'Ervilha':           'leguminosas',
  'Batata':            'hortalicas',   'Mandioca':          'hortalicas',
  'Cebola':            'hortalicas',   'Alho':              'hortalicas',
  'Tomate':            'hortalicas',   'Abobora':           'hortalicas',
  'Cenoura':           'hortalicas',   'Couve-flor':        'hortalicas',
  'Pepino':            'hortalicas',   'Pimentao':          'hortalicas',
  'Repolho':           'hortalicas',   'Beterraba':         'hortalicas',
  'Chuchu':            'hortalicas',   'Brocolis':          'hortalicas',
  'Abobrinha':         'hortalicas',   'Alface':            'hortalicas',
  'Berinjela':         'hortalicas',
  'Melancia':          'frutas',       'Goiaba':            'frutas',
  'Laranja':           'frutas',       'Tangerina':         'frutas',
  'Melao':             'frutas',       'Uva':               'frutas',
  'Ameixa':            'frutas',       'Kiwi':              'frutas',
  'Maracuja':          'frutas',       'Nectarina':         'frutas',
  'Pessego':           'frutas',       'Pera':              'frutas',
  'Caqui':             'frutas',       'Maca':              'frutas',
  'Banana':            'frutas',       'Abacate':           'frutas',
  'Cacau':             'frutas',       'Figo':              'frutas',
  'Manga':             'frutas',       'Mamao':             'frutas',
  'Morango':           'frutas',       'Lichia':            'frutas',
  'Abacaxi':           'frutas',       'Limao':             'frutas',
  'Lima':              'frutas',       'Graviola':          'frutas',
  'Atemoia':           'frutas',
  'Pastagem':          'outros',       'Floresta':          'outros',
}

# -----------------------------------------------------------------------------
# Correção de nomes de municípios
# Nomes com grafia incorreta/obsoleta no SISSER → nome oficial (sem acento)
# Baseado em: Patrícia Siqueira, Walef Santos e Leonardo (dados até 2022) +
#             Wellinton Santos (dados de 2023)
# -----------------------------------------------------------------------------

REPLACERS_MUN = {
  'pereirinhas': 'desterro_de_entre_rios', 'torrinhas': 'pinheiro_machado',
  'cerro_do_roque': 'butia', 'pinheiro_marcado': 'carazinho',
  'ourilandia': 'barbosa_ferraz', 'nova_brasilia': 'araruna',
  'trentin': 'jaboticaba', 'vale_veneto': 'sao_joao_do_polesine',
  'sao_pedro_tobias': 'dionisio_cerqueira', 'vale_formoso': 'novo_horizonte',
  'cavalheiro': 'ipameri', 'rio_do_salto': 'cascavel',
  'cascata': 'pelotas', 'gramadinho': 'itapetininga',
  'cavajureta': 'sao_vicente_do_sul', 'conceicao_de_monte_alegre': 'paraguacu_paulista',
  'taquarichim': 'jaguari', 'tres_placas': 'tapejara',
  'passinhos': 'osorio', 'sede_alvorada': 'cascavel',
  'juliania': 'herculandia', 'basilio': 'herval',
  'esperanca_do_norte': 'alvorada_do_sul', 'itaboa': 'ribeirao_branco',
  'alto_alvorada': 'orizona', 'jafa': 'garca',
  'itahum': 'dourados', 'arapuan': 'arapua',
  'rio_do_mato': 'francisco_beltrao', 'nossa_senhora_da_candelaria': 'bandeirantes',
  'sarandira': 'juiz_de_fora', 'plano_alto': 'uruguaiana',
  'rocas_novas': 'caete', 'frei_timoteo': 'jataizinho',
  'vidigal': 'cianorte', 'colonia_esperanca': 'arapongas',
  'cristo_rei': 'capanema', 'graccho_cardoso': 'gracho_cardoso',
  'marajo': 'nova_aurora', 'valerio': 'planalto',
  'sao_camilo': 'palotina', 'triolandia': 'ribeirao_do_pinhal',
  'ibare': 'lavras_do_sul', 'bocaja': 'douradina',
  'itao': 'itaqui', 'vila_gandhi': 'primeiro_de_maio',
  'honoropolis': 'campina_verde', 'colonia_sao_joao': 'cruz_alta',
  'otavio_rocha': 'flores_da_cunha', 'engenheiro_maia': 'itabera',
  'hidraulica': 'pelotas', 'palmitopolis': 'nova_aurora',
  'biritiba_ussu': 'mogi_das_cruzes', 'trevo_do_jose_rosario': 'leopoldo_de_bulhoes',
  'poema': 'nova_tebas', 'espigao': 'regente_feijo',
  'irere': 'londrina', 'bairro_limoeiro': 'londrina',
  'capao_grande': 'muitos_capoes', 'santo_antonio_do_paranapanema': 'candido_mota',
  'rincao_do_cristovao_pereira': 'mostardas', 'sao_luiz_do_oeste': 'toledo',
  'colonia_socorro': 'guarapuava', 'colonia_vitoria': 'guarapuava',
  'vale_dos_vinhedos': 'bento_goncalves', 'barro_vermelho': 'gravatai',
  'santo_antonio_do_rio_verde': 'catalao', 'nova_lourdes': 'sao_joao',
  'santa_cruz_do_timbo': 'porto_uniao', 'santauta': 'camaqua',
  'guaragi': 'ponta_grossa', 'caetano_mendes': 'tibagi',
  'torquato_severo': 'dom_pedrito', 'pontoes': 'afonso_claudio',
  'ivailandia': 'engenheiro_beltrao', 'capao_seco': 'sidrolandia',
  'aparecida_do_oeste': 'tuneiras_do_oeste', 'engenheiro_luiz_englert': 'sertao',
  'bela_vista_do_sul': 'mafra', 'gamadinho': 'cascavel',
  'juruce': 'jardinopolis', 'novo_diamantino': 'diamantino',
  'paranagi': 'sertaneja', 'roda_velha': 'sao_desiderio',
  'ferreira': 'cachoeira_do_sul', 'mirante_do_piquiri': 'alto_piquiri',
  'bentopolis': 'nossa_senhora_das_gracas', 'passo_liso': 'laranjeiras_do_sul',
  'saica': 'cacequi', 'indapolis': 'dourados',
  'taim': 'rio_grande', 'correia_de_freitas': 'apucarana',
  'iolopolis': 'sao_jorge_doeste', 'lajeado_cerne': 'santo_angelo',
  'itapocu': 'araquari', 'cordilheira': 'cachoeira_do_sul',
  'colonia_castrolanda': 'castro', 'capivarita': 'rio_pardo',
  'nossa_senhora_de_caravaggio': 'nova_veneza', 'candia': 'pontal',
  'herveira': 'campina_da_lagoa', 'santa_flora': 'santa_maria',
  'santa_lucia_do_piai': 'caxias_do_sul', 'figueira_do_oeste': 'engenheiro_beltrao',
  'porto_mendes': 'marechal_candido_rondon', 'holambra_ii': 'paranapanema',
  'perola_independente': 'maripa', 'alto_santa_fe': 'nova_santa_rosa',
  'calcilandia': 'goias', 'comandai': 'santo_angelo',
  'cerrito_alegre': 'pelotas', 'fazenda_jangada': 'cascavel',
  'guajuvira': 'araucaria', 'guacu': 'dourados',
  'agua_azul': 'lapa', 'barra_grande': 'itapejara_doeste',
  'colonia_samambaia': 'guarapuava', 'encantado_doeste': 'assis_chateaubriand',
  'pulinopolis': 'mandaguacu', 'piquirivai': 'campo_mourao',
  'bateias': 'campo_largo', 'novo_sarandi': 'toledo',
  'nova_sardenha': 'farroupilha', 'mariental': 'lapa',
  'arroio_do_so': 'santa_maria', 'warta': 'londrina',
  'vila_marques': 'aral_moreira', 'nova_concordia': 'francisco_beltrao',
  'piriquitos': 'ponta_grossa', 'vila_ipiranga': 'toledo',
  'doutor_oliveira_castro': 'guaira', 'colonia_z_3': 'pelotas',
  'novo_sobradinho': 'toledo', 'vila_diniz': 'cruzmaltina',
  'catanduvas_do_sul': 'contenda', 'jansen': 'farroupilha',
  'polo_petroquimico_de_triunfo': 'triunfo', 'carumbe': 'itapora',
  'sao_joao_doeste': 'cascavel', 'albardao': 'rio_pardo',
  'banhado_do_colegio': 'camaqua', 'bacupari': 'palmares_do_sul',
  'souza_ramos': 'getulio_vargas', 'montese': 'itapora',
  'rincao_dos_mendes': 'santo_angelo', 'batatuba': 'piracaia',
  'vila_oliva': 'caxias_do_sul', 'rincao_del_rei': 'rio_pardo',
  'turiba_do_sul': 'itabera', 'aracaiba': 'apiai',
  'macaia': 'bom_sucesso', 'criuva': 'caxias_do_sul',
  'bom_sucesso_de_patos': 'patos_de_minas', 'amparo_da_serra': 'amparo_do_serra',
  'arace': 'domingos_martins', 'agisse': 'rancharia',
  'perico': 'sao_joaquim', 'capao_da_porteira': 'viamao',
  'gardenia': 'rancharia', 'couto_de_magalhaes': 'couto_magalhaes',
  'fazenda_souza': 'caxias_do_sul', 'cardeal': 'elias_fausto',
  'sitio_grande': 'sao_desiderio', 'sao_valerio_da_natividade': 'sao_valerio',
  'capao_da_lagoa': 'guarapuava', 'santa_rita_do_ibitipoca': 'santa_rita_de_ibitipoca',
  'ipiabas': 'barra_do_pirai', 'charco': 'castro',
  'dez_de_maio': 'toledo', 'jubai': 'conquista',
  'aquidaban': 'marialva', 'bragantina': 'assis_chateaubriand',
  'guaravera': 'londrina', 'ibitiuva': 'pitangueiras',
  'indios': 'lages', 'tres_bicos': 'candido_de_abreu',
  'santa_rita_do_oeste': 'santa_rita_doeste', 'vila_nova': 'toledo',
  'lerroville': 'londrina', 'panema': 'santa_mariana',
  'santo_antonio_dos_campos': 'divinopolis', 'vila_seca': 'caxias_do_sul',
  'frutal_do_campo': 'candido_mota', 'colonia_centenario': 'cascavel',
  'garibaldina': 'garibaldi', 'macucos': 'getulina',
  'siqueira_belo': 'barracao', 'sao_joao_dos_mellos': 'julio_de_castilhos',
  'crispim_jaques': 'teofilo_otoni', 'cruzaltina': 'douradina',
  'bacuriti': 'cafelandia', 'coronel_prestes': 'encruzilhada_do_sul',
  'sao_bento_baixo': 'nova_veneza', 'atiacu': 'sarandi',
  'pacheca': 'camaqua', 'bom_jardim_do_sul': 'ivai',
  'pampeiro': 'santana_do_livramento', 'quinzopolis': 'santa_mariana',
  'bonfim_paulista': 'ribeirao_preto', 'uvaia': 'ponta_grossa',
  'itororo_do_paranapanema': 'pirapozinho', 'yolanda': 'ubirata',
  'nova_altamira': 'faxinal', 'rincao_do_meio': 'sao_tome',
  'rincao_doce': 'santo_antonio_do_planalto', 'paiquere': 'londrina',
  'ouroana': 'rio_verde', 'abapa': 'castro',
  'calogeras': 'arapoti', 'alexandrita': 'iturama',
  'campo_do_bugre': 'rio_bonito_do_iguacu', 'barao_de_lucena': 'nova_esperanca',
  'porteira_preta': 'fenix', 'sao_jose_da_reserva': 'santa_cruz_do_sul',
  'sao_luiz_do_puruna': 'balsa_nova', 'dorizon': 'mallet',
  'bernardelli': 'rondon', 'lagoa_do_bauzinho': 'rio_verde',
  'nova_cardoso': 'itajobi', 'bela_vista_do_piquiri': 'campina_da_lagoa',
  'nossa_senhora_da_aparecida': 'rolandia', 'monte_alverne': 'santa_cruz_do_sul',
  'azevedo_sodre': 'sao_gabriel', 'sao_joaquim_do_pontal': 'itambaraca',
  'bourbonia': 'barbosa_ferraz', 'guaraciaba_doeste': 'tupi_paulista',
  'colonia_melissa': 'cascavel', 'selva': 'londrina',
  'cabeceira_do_apa': 'ponta_pora', 'cachoeira_de_emas': 'pirassununga',
  'barragem_do_itu': 'macambara', 'taquaruna': 'londrina',
  'sede_progresso': 'francisco_beltrao', 'porto_vilma': 'deodapolis',
  'irui': 'rio_pardo', 'novo_tres_passos': 'marechal_candido_rondon',
  'tereza_breda': 'barbosa_ferraz', 'guaipora': 'cafezal_do_sul',
  'vida_nova': 'sapopema', 'fazenda_colorado': 'fortaleza_dos_valos',
  'conselheiro_zacarias': 'santo_antonio_da_platina', 'palmira': 'sao_joao_do_triunfo',
  'capivara': 'erval_seco', 'nova_patria': 'presidente_bernardes',
  'espinilho_grande': 'tupancireta', 'aguas_claras': 'viamao',
  'santa_rita_da_floresta': 'cantagalo', 'papagaios_novos': 'palmeira',
  'passo_real': 'salto_do_jacui', 'triangulo': 'engenheiro_beltrao',
  'capela_sao_paulo': 'sao_luiz_gonzaga', 'nova_casa_verde': 'nova_andradina',
  'curral_alto': 'santa_vitoria_do_palmar', 'ipomeia': 'rio_das_antas',
  'tapinas': 'itapolis', 'vassoural': 'ibaiti',
  'cachoeira_do_espirito_santo': 'ribeirao_claro', 'picadinha': 'dourados',
  'palmeirinha': 'guarapuava', 'passo_do_verde': 'sao_sepe',
  'alto_do_amparo': 'tibagi', 'jacarandira': 'resende_costa',
  'guarapua': 'dois_corregos', 'pedra_branca_de_itarare': 'itarare',
  'nova_milano': 'farroupilha', 'rio_toldo': 'getulio_vargas',
  'juvinopolis': 'cascavel', 'granja_getulio_vargas': 'palmares_do_sul',
  'porto_santana': 'porto_barreiro', 'coxilha_rica': 'itapejara_doeste',
  'vila_freire': 'cerrito', 'bonfim_da_feira': 'feira_de_santana',
  'mimoso': 'barao_de_melgaco', 'felpudo': 'campo_largo',
  'pulador': 'passo_fundo', 'bau': 'candiota',
  'tupinamba': 'astorga', 'jazidas': 'formigueiro',
  'mariza': 'sao_pedro_do_ivai', 'patrocinio_de_caratinga': 'caratinga',
  'campo_seco': 'rosario_do_sul', 'sao_miguel_do_cambui': 'marialva',
  'pau_dalho_do_sul': 'assai', 'conciolandia': 'perola_doeste',
  'margarida': 'marechal_candido_rondon', 'concordia_do_oeste': 'toledo',
  'daltro_filho': 'imigrante', 'dario_lassance': 'candiota',
  'amandina': 'ivinhema', 'guarapuavinha': 'inacio_martins',
  'vila_nova_de_florenca': 'sao_jeronimo_da_serra', 'geremia_lunardelli': 'nova_cantu',
  'riverlandia': 'rio_verde', 'sao_jose_das_laranjeiras': 'maracai',
  'fluviopolis': 'sao_mateus_do_sul', 'cerrito_do_ouro': 'sao_sepe',
  'juciara': 'kalore', 'colonia_jordaozinho': 'guarapuava',
  'barra_dourada': 'neves_paulista', 'sao_clemente': 'santa_helena',
  'santa_cruz_da_estrela': 'santa_rita_do_passa_quatro', 'pedro_lustosa': 'reserva_do_iguacu',
  'guaipava': 'paraguacu', 'bexiga': 'rio_pardo',
  'boca_do_monte': 'santa_maria', 'tupi_silveira': 'candiota',
  'iguipora': 'marechal_candido_rondon', 'passo_das_pedras': 'capao_do_leao',
  'capane': 'cachoeira_do_sul', 'jangada_do_sul': 'general_carneiro',
  'malu': 'terra_boa', 'esquina_piratini': 'bossoroca',
  'caraja': 'jesuitas', 'santo_antonio_do_palmital': 'rio_bom',
  'joao_arregui': 'uruguaiana', 'clemente_argolo': 'lagoa_vermelha',
  'alto_da_uniao': 'ijui', 'fernao_dias': 'munhoz_de_melo',
  'taquara_verde': 'cacador', 'apiaba': 'imbituva',
  'ponte_vermelha': 'sao_gabriel_do_oeste', 'floropolis': 'paranacity',
  'apiai_mirim': 'capao_bonito', 'jacipora': 'dracena',
  'silveira': 'sao_jose_dos_ausentes', 'piquiri': 'nova_esperanca_do_sul',
  'prudencio_e_moraes': 'general_salgado', 'ibiporanga': 'tanabi',
  'saltinho_do_oeste': 'alto_piquiri', 'guardinha': 'sao_sebastiao_do_paraiso',
  'bom_retiro_da_esperanca': 'angatuba', 'ouro_verde_do_piquiri': 'corbelia',
  'campina_de_fora': 'ribeirao_branco', 'santa_esmeralda': 'santa_cruz_de_monte_castelo',
  'cambaratiba': 'ibitinga', 'romeopolis': 'arapua',
  'clarinia': 'santa_cruz_do_rio_pardo', 'tres_vendas': 'cachoeira_do_sul',
  'candeia': 'maripa', 'joca_tavares': 'bage',
  'veredas': 'joao_pinheiro', 'lageado_de_aracaiba': 'apiai',
  'guarizinho': 'itapeva', 'santa_fe_do_pirapo': 'marialva',
  'santa_izabel': 'sao_joaquim', 'vila_formosa': 'dourados',
  'rincao_comprido': 'augusto_pestana', 'espigao_do_oeste': 'espigao_doeste',
  'tres_capoes': 'guarapuava', 'bandeirantes_doeste': 'formosa_do_oeste',
  'jurupema': 'taquaritinga', 'covo': 'mangueirinha',
  'parana_doeste': 'moreira_sales', 'sao_francisco_de_imbau': 'congonhinhas',
  'jaracatia': 'goioere', 'barreiro': 'ijui',
  'colonia_cachoeira': 'guarapuava', 'arvore_grande': 'paranaiba',
  'vila_vargas': 'dourados', 'ubauna': 'sao_joao_do_ivai',
  'ibiaci': 'primeiro_de_maio', 'aparecida_de_minas': 'frutal',
  'retiro_grande': 'campo_largo', 'pedras': 'lapa',
  'campo_lindo': 'campo_limpo_de_goias', 'lagoa_branca': 'casa_branca',
  'herval_grande': 'laranjeiras_do_sul', 'boa_vista_de_santa_maria': 'unai',
  'leao': 'campos_novos', 'marilu': 'mariluz',
  'ilha_dos_marinheiros': 'rio_grande', 'barra_santa_salete': 'manoel_ribas',
  'santa_luzia_da_alvorada': 'sao_joao_do_ivai', 'amoras': 'taquari',
  'colonia_medeiros': 'independencia', 'esteios': 'luz',
  'tronco': 'castro', 'angai': 'fernandes_pinheiro',
  'forte': 'sao_joao_dalianca', 'cachoeira_do_mato': 'teixeira_de_freitas',
  'sao_jose_do_iguacu': 'sao_miguel_do_iguacu', 'gamela': 'eneas_marques',
  'posse_dos_linhares': 'rio_dos_indios', 'sao_miguel_do_cajuru': 'sao_joao_del_rei',
  'jardinesia': 'prata', 'canabrava': 'joao_pinheiro',
  'salles_de_oliveira': 'campina_da_lagoa', 'amandaba': 'mirandopolis',
  'ibiranhem': 'mucuri', 'capo_ere': 'campo_ere',
  'sanatorio_santa_fe': 'tres_coracoes', 'curupa': 'tabatinga',
  'sao_jose_do_ivai': 'santa_isabel_do_ivai', 'garapuava': 'unai',
  'capao_rico': 'guarapuava', 'taquarucu': 'palmas',
  'guarauna': 'teixeira_soares', 'merces_de_agua_limpa': 'sao_tiago',
  'albuquerque': 'corumba', 'pranchada': 'doutor_mauricio_cardoso',
  'presidente_castelo': 'deodapolis', 'bom_plano': 'vista_gaucha',
  'debrasa': 'brasilandia', 'rio_claro_do_sul': 'mallet',
  'rincao_do_appel': 'pinhal_grande', 'serra_bonita': 'buritis',
  'vargeado': 'nova_trento', 'cachoeira_de_santa_cruz': 'vicosa',
  'guaicui': 'varzea_da_palma', 'marcorama': 'garibaldi',
  'rio_pardinho': 'santa_cruz_do_sul', 'bairro_cachoeira': 'sao_jose_dos_pinhais',
  'lagolandia': 'pirenopolis', 'alcantilado': 'guiratinga',
  'aparecida_de_sao_manuel': 'sao_manuel', 'girassol': 'cocalzinho_de_goias',
  'santa_izabel_do_sul': 'arroio_grande', 'douradilho': 'barra_do_ribeiro',
  'coronel_goulart': 'alvares_machado', 'barra_do_brejo': 'bom_conselho',
  'cisneiros': 'palma', 'rio_preto_do_sul': 'mafra',
  'guariroba': 'taquaritinga', 'atafona': 'santo_angelo',
  'lavouras': 'alto_paraguai', 'rincao_dos_mellos': 'girua',
  'santa_eudoxia': 'sao_carlos', 'corrego_moacyr_avidos': 'governador_lindenberg',
  'santa_terezinha_do_salto': 'lages', 'cachoeira_da_serra': 'altamira',
  'linha_bonita': 'cerro_largo', 'bandeirinha': 'camaqua',
  'baixa': 'uberaba', 'canastrao': 'tiros',
  'argenita': 'ibia', 'barra_seca': 'sao_mateus',
  'coxilha_grande': 'vacaria', 'conceicao_da_brejauba': 'gonzaga',
  'acungui': 'rio_branco_do_sul', 'roberto': 'pindorama',
  'catucaba': 'sao_gabriel', 'sao_jose_da_gloria': 'victor_graeff',
  'manchinha': 'tres_de_maio', 'divino_espirito_santo': 'alterosa',
  'taquari_dos_russos': 'ponta_grossa', 'bom_principio_do_oeste': 'toledo',
  'rio_da_prata': 'nova_laranjeiras', 'olho_agudo': 'sao_jose_dos_pinhais',
  'algodoes': 'quijingue', 'comur': 'planaltina_do_parana',
  'botucarai': 'candelaria', 'pangare': 'quitandinha',
  'antonio_kerpel': 'coronel_bicaco', 'monte_bonito': 'pelotas',
  'aparecida_do_ivai': 'santa_monica', 'piao': 'santa_rita_de_caldas',
  'rincao_dos_meoti': 'santo_angelo', 'cerro_alegre_baixo': 'santa_cruz_do_sul',
  'ribeirao_de_sao_domingos': 'santa_margarida', 'borman': 'guaraniacu',
  'toroqua': 'sao_francisco_de_assis', 'paiolinho': 'poco_fundo',
  'joa': 'joaquim_tavora', 'conceicao_de_minas': 'dionisio',
  'caravagio': 'sorriso', 'galena': 'presidente_olegario',
  'conceicao_do_muqui': 'mimoso_do_sul', 'piracaiba': 'araguari',
  'avelino_paranhos': 'espumoso', 'alfredo_brenner': 'ibiruba',
  'tapui': 'toledo', 'bezerra': 'formosa',
  'mirim': 'severiano_de_almeida', 'bom_recreio': 'passo_fundo',
  'nhu_pora': 'sao_borja', 'vila_nelita': 'agua_doce_do_norte',
  'luar': 'sao_joao_do_ivai', 'nossa_senhora_de_fatima': 'santa_maria',
  'nova_cultura': 'papanduva', 'doutor_ernesto': 'toledo',
  'eduardo_xavier_da_silva': 'jaguariaiva', 'anhandui': 'campo_grande',
  'colonia_municipal': 'santo_angelo', 'linha_vitoria': 'almirante_tamandare_do_sul',
  'irakitan': 'tangara', 'engenho_grande': 'agua_santa',
  'mariquita': 'tabocas_do_brejo_velho', 'claro_de_minas': 'vazante',
  'sanga_puita': 'ponta_pora', 'linha_gloria': 'lagoa_dos_tres_cantos',
  'auriverde': 'crixas', 'goiaminas': 'formoso',
  'novo_cravinhos': 'pompeia', 'sede_aurora': 'quinze_de_novembro',
  'cavera': 'rosario_do_sul', 'sao_sebastiao_da_vala': 'aimores',
  'marari': 'tangara', 'caburu': 'sao_joao_del_rei',
  'braco_forte': 'tenente_portela', 'batista_botelho': 'oleo',
  'marcondesia': 'monte_azul_paulista', 'catole': 'campina_grande',
  'garrafao': 'santa_maria_de_jetiba', 'esquina_gaucha': 'entre_ijuis',
  'campinal': 'presidente_epitacio', 'ipuca': 'sao_fidelis',
  'borore': 'macambara', 'avai_do_jacinto': 'jacinto',
  'batovi': 'sao_gabriel', 'bosque': 'cachoeira_do_sul',
  'estacao_roca_nova': 'piraquara', 'cafemirim': 'tarumirim',
  'sao_jose_dos_campos_borges': 'campos_borges', 'francisco_frederico_teixeira_guimaraes': 'palmas',
  'ernesto_alves': 'santiago', 'sede_independencia': 'passo_fundo',
  'assarai': 'pocrane', 'santo_antonio_do_manhuacu': 'caratinga',
  'tabajara': 'machadinho_doeste', 'caita': 'sao_mateus_do_sul',
  'aguas_de_contendas': 'conceicao_do_rio_verde', 'giruazinho': 'senador_salgado_filho',
  'bendego': 'canudos', 'djalma_coutinho': 'santa_leopoldina',
  'santo_agostinho': 'agua_doce_do_norte', 'ruralminas': 'unai',
  'itabatan': 'mucuri', 'culturama': 'fatima_do_sul',
  'claudinapolis': 'nazario', 'quaraim': 'tres_de_maio',
  'sao_sebastiao_da_vitoria': 'sao_joao_del_rei', 'rincao_de_sao_miguel': 'alegrete',
  'brugnarotto': 'sao_jose_do_ouro', 'cerro_chato': 'herval',
  'sao_joao_do_pinhal': 'sao_jeronimo_da_serra', 'matarazzo': 'pedro_osorio',
  'caetano_lopes': 'jeceaba', 'capela_da_luz': 'monte_alegre_dos_campos',
  'marimbondo': 'siqueira_campos', 'cotaxe': 'ecoporanga',
  'cerro_claro': 'sao_pedro_do_sul', 'itaiacoca': 'ponta_grossa',
  'olimpio_campos': 'sao_joao_da_ponte', 'felipe_schmidt': 'canoinhas',
  'padre_gonzales': 'tres_passos', 'centralito': 'cascavel',
  'coronel_teixeira': 'marcelino_ramos', 'vila_rica_do_ivai': 'icaraima',
  'campo_limpo': 'campo_limpo_de_goias', 'pontalete': 'tres_pontas',
  'tupantuba': 'santiago', 'ligacao_do_para': 'dom_eliseu',
  'botelho': 'santa_adelia', 'afonso_rodrigues': 'sao_luiz_gonzaga',
  'bel_rios': 'diamantino', 'santo_antonio_da_esperanca': 'santa_cruz_de_goias',
  'bueno_de_andrada': 'araraquara', 'aroeira': 'cascavel',
  'cruzaltinha': 'agua_santa', 'caraiba_do_norte': 'sao_francisco_do_maranhao',
  'chumbo': 'patos_de_minas', 'arraial_dajuda': 'porto_seguro',
  'bananas': 'nova_laranjeiras', 'flor_da_serra': 'flor_da_serra_do_sul',
  'eldorado_dos_carajas': 'curionopolis', 'bom_jesus_do_divino': 'divino',
  'espigao_alto': 'barracao', 'agua_vermelha': 'sao_carlos',
  'ingas': 'nova_granada', 'cruzeiro_do_norte': 'bonopolis',
  'forninho': 'cacapava_do_sul', 'lagoa_bonita': 'deodapolis',
  'braco_do_rio': 'conceicao_da_barra', 'agulha': 'fernando_prestes',
  'arruda': 'rosario_oeste', 'chaveslandia': 'santa_vitoria',
  'costa_da_cadeia': 'triunfo', 'bemposta': 'tres_rios',
  'vera_guarani': 'paulo_frontin', 'itapua': 'viamao',
  'barreirinho': 'sarandi', 'barcelos_do_sul': 'camamu',
  'santo_antonio_de_barcelona': 'caravelas', 'santa_cruz_da_conceic?o': 'santa_cruz_da_conceicao',
  'monte_sinai': 'barra_de_sao_francisco', 'indaia_grande': 'tres_lagoas',
  'marcianopolis': 'santo_antonio_do_sudoeste', 'vila_campos': 'tapejara',
  'marrecas': 'francisco_beltrao', 'colonia_general_carneiro': 'general_carneiro',
  'santa_cruz_das_lajes': 'santo_antonio_da_barra', 'frei_sebastiao': 'palmares_do_sul',
  'pires_belo': 'catalao', 'banhado': 'piraquara',
  'fortaleza_do_tabocao': 'tabocao', 'indai': 'mundo_novo',
  'lagoa_azul': 'cubatao', 'planalto_do_sul': 'teodoro_sampaio',
  'ascencao': 'para_de_minas', 'nova_caceres': 'caceres',
  'igara': 'senhor_do_bonfim', 'ressaca_do_buriti': 'santo_angelo',
  'arlindo': 'venancio_aires', 'bote': 'herval',
  'aurora_do_iguacu': 'sao_miguel_do_iguacu', 'alto_capim': 'aimores',
  'rincao_dos_roratos': 'sete_de_setembro', 'passo_manso': 'blumenau',
  'clara': 'mata', 'fazenda_da_estrela': 'vacaria',
  'entrepelado': 'taquara', 'graciosa': 'paranavai',
  'rio_telha': 'ibiaca', 'itamira': 'apora',
  'guarapiranga': 'sao_paulo', 'alvacao': 'coracao_de_jesus',
  'alto_sao_joao': 'roncador', 'vinhatico': 'montanha',
  'linha_ocidental': 'arroio_do_tigre', 'aparecida_do_bonito': 'santa_rita_doeste',
  'macuco_de_minas': 'itumirim', 'jaguaritira': 'malacacheta',
  'pequia': 'iuna', 'cadeadinho': 'irati',
  'arvore_so': 'santa_vitoria_do_palmar', 'bueno': 'bueno_brandao',
  'douradinho': 'machado', 'geriacu': 'uruacu',
  'cupins': 'aparecida_do_taboado', 'paraiso_do_leste': 'poxoreu',
  'garcias': 'tres_lagoas', 'martinesia': 'uberlandia',
  'cajui': 'sento_se', 'cazuza_ferreira': 'sao_francisco_de_paula',
  'bom_jesus_do_querendo': 'natividade', 'sao_joao_de_itaguacu': 'urupes',
  'costa_machado': 'mirante_do_paranapanema', 'tecainda': 'martinopolis',
  'esquina_ipiranga': 'senador_salgado_filho', 'arrozal': 'pirai',
  'araguaia': 'marechal_floriano', 'engenheiro_balduino': 'monte_aprazivel',
  'faxinal_preto': 'sao_jose_dos_ausentes', 'cafeeiros': 'cruzeiro_do_oeste',
  'vista_nova': 'crissiumal', 'capivari_da_mata': 'ituverava',
  'campina': 'campina_grande_do_sul', 'ariri': 'cananeia',
  'igarai': 'mococa', 'lejeado_micuim': 'santo_angelo',
  'bicuiba': 'raul_soares', 'aparecida_de_monte_alto': 'monte_alto',
  'tindiquera': 'araucaria', 'pedro_paiva': 'santo_augusto',
  'santo_rei': 'nova_cantu', 'bom_sera': 'cristal',
  'cintra_pimentel': 'nova_londrina', 'jaciaba': 'prudentopolis',
  'catanduva_grande': 'santo_antonio_da_patrulha', 'caldas_do_jorro': 'tucano',
  'ariau': 'barreirinha', 'chorao': 'ijui',
  'aracacu': 'buri', 'taveira': 'niquelandia',
  'suspiro': 'sao_gabriel', 'araxas': 'presidente_bernardes',
  'carnaiba_do_sertao': 'juazeiro', 'bela_vista_do_ivai': 'fenix',
  'rio_tigre': 'sananduva', 'sao_gabriel_de_goias': 'planaltina',
  'xiniqua': 'sao_pedro_do_sul', 'doutor_edgardo_pereira_velho': 'mostardas',
  'rosalandia': 'sao_luis_de_montes_belos', 'amanhece': 'araguari',
  'pana': 'nova_alvorada_do_sul', 'desemboque': 'sacramento',
  'guapiranga': 'lins', 'jacare': 'barra_do_jacare',
  'luzimangues': 'porto_nacional', 'morello': 'governador_lindenberg',
  'itaio': 'itaiopolis', 'emboabas': 'sao_joao_del_rei',
  'alberto_moreira': 'barretos', 'posselandia': 'guapo',
  'grapia': 'paraiso', 'cerro_alto': 'tuparendi',
  'guachos': 'martinopolis', 'cocaes': 'sarapui',
  'baguacu': 'olimpia', 'jorge_lacerda': 'dionisio_cerqueira',
  'porto_camargo': 'icaraima', 'bezerro_branco': 'caceres',
  'sao_jorge_do_tiradentes': 'rio_bananal', 'faxinal_do_ceu': 'pinhao',
  'baia_alta': 'ponte_serrada', 'cruzeiro_dos_peixotos': 'uberlandia',
  'alto_recreio': 'ronda_alta', 'pratos': 'novo_machado',
  'colonia_das_almas': 'catuipe', 'bom_fim': 'jaraguari',
  'socavao': 'castro', 'baus': 'costa_rica',
  'pacotuba': 'cachoeiro_de_itapemirim', 'cinquentenario': 'tuparendi',
  'duplo_ceu': 'palestina', 'cerro_do_ouro': 'sao_gabriel',
  'aguacu': 'cuiaba', 'agua_branca': 'guarupuava',
  'almeidas': 'conselheiro_lafaiete', 'altaneira': 'maringa',
  'amparo_de_sao_francisco': 'amparo_do_sao_francisco', 'aricanduva': 'arapongas',
  'arquimedes': 'cascavel', 'barreiro_do_rio_verde': 'verdelandia',
  'barro_preto': 'ventania', 'bocaina': 'ponta_grossa',
  'catimbau': 'alegrete', 'espirito_santo': 'londrina',
  'gaucha': 'cidade_gaucha', 'inhandui': 'alegrete',
  'linha_giacomini': 'toledo', 'ouro_preto': 'toledo',
  'primavera': 'rosana', 'santa_efigenia': 'santa_efigenia_de minas',
  'santo_antonio_do_leverger': 'santo_antonio_de_leverger', 'sape': 'caldas_novas',
  'sarandi': 'itumbiara', 'tuiuti': 'bento_goncalves',
}

# -----------------------------------------------------------------------------
# Mapeamento de código numérico IBGE de UF → sigla
# -----------------------------------------------------------------------------

UF_MAP = {
  12: 'AC', 27: 'AL', 13: 'AM', 16: 'AP', 29: 'BA', 23: 'CE', 53: 'DF',
  32: 'ES', 52: 'GO', 21: 'MA', 31: 'MG', 50: 'MS', 51: 'MT', 15: 'PA',
  25: 'PB', 26: 'PE', 22: 'PI', 41: 'PR', 33: 'RJ', 24: 'RN', 11: 'RO',
  14: 'RR', 43: 'RS', 42: 'SC', 28: 'SE', 35: 'SP', 17: 'TO',
}

# -----------------------------------------------------------------------------
# Normalização dos nomes de seguradoras
# -----------------------------------------------------------------------------

REPLACERS_SEG = {
  'Aliança do Brasil Seguros S/A.':                'alianca',
  'Allianz Seguros S.A':                           'allianz',
  'BRASILSEG COMPANHIA DE SEGUROS':                'brasilseg',
  'Companhia Excelsior de Seguros':                'excelsior',
  'Essor Seguros S.A.':                            'essor',
  'FairFax Brasil Seguros Corporativos S/A':       'fairfax',
  'Mapfre Seguros Gerais S.A.':                    'mapfre',
  'Newe Seguros S.A':                              'newe',
  'Porto Seguro Companhia de Seguros Gerais':      'porto_seguro',
  'Sancor Seguros do Brasil S.A.':                 'sancor',
  'Sompo Seguros S/A':                             'sompo',
  'Swiss Re Corporate Solutions Brasil S.A.':      'swiss',
  'Tokio Marine Seguradora S.A.':                  'tokio_marine',
  'Too Seguros S.A.':                              'too',
  'Sombrero Seguros S/A':                          'sombrero',
}

# -----------------------------------------------------------------------------
# Mapeamento de UF → macrorregião brasileira
# -----------------------------------------------------------------------------

REGIAO_MAP = {
  'TO': 'Norte',      'AM': 'Norte',      'AC': 'Norte',
  'RR': 'Norte',      'PA': 'Norte',      'AP': 'Norte',      'RO': 'Norte',
  'CE': 'Nordeste',   'BA': 'Nordeste',   'RN': 'Nordeste',
  'SE': 'Nordeste',   'MA': 'Nordeste',   'PI': 'Nordeste',
  'PB': 'Nordeste',   'AL': 'Nordeste',   'PE': 'Nordeste',
  'SP': 'Sudeste',    'MG': 'Sudeste',    'ES': 'Sudeste',    'RJ': 'Sudeste',
  'GO': 'Centro-Oeste', 'MT': 'Centro-Oeste',
  'MS': 'Centro-Oeste', 'DF': 'Centro-Oeste',
  'PR': 'Sul',        'SC': 'Sul',        'RS': 'Sul',
}

# -----------------------------------------------------------------------------
# Ordem canônica das colunas na camada Silver
# (colunas ausentes são ignoradas automaticamente no pipeline)
# -----------------------------------------------------------------------------

COLUNAS_FINAIS = [
  'apolice', 'dt_inicio_vigencia', 'dt_fim_vigencia',
  'mun', 'nome_mun', 'uf', 'regiao', 'lat', 'lon',
  'seguradora', 'tipo', 'cultura', 'tipo_cultura',
  'area', 'animal', 'duracao',
  'prod_est', 'prod_seg', 'nivel_cob', 'total_seg', 'premio',
  'taxa', 'subvencao', 'indenizacao', 'evento', 'sinistro', 'sinistralidade',
]

# -----------------------------------------------------------------------------
# Separação de colunas para a camada Gold (Feature Store)
# COLUNAS_FEATURES: preditores — sem variáveis de desfecho pós-contratual
# COLUNAS_LABELS:   variáveis resposta + chaves de junção
# -----------------------------------------------------------------------------

COLUNAS_FEATURES = [
  'apolice', 'dt_inicio_vigencia', 'dt_fim_vigencia',
  'mun', 'nome_mun', 'uf', 'regiao', 'lat', 'lon',
  'seguradora', 'tipo', 'cultura', 'tipo_cultura',
  'area', 'animal', 'duracao',
  'prod_est', 'prod_seg', 'nivel_cob', 'total_seg', 'premio',
  'taxa', 'subvencao',
]

COLUNAS_LABELS = [
  'apolice', 'dt_inicio_vigencia',
  'evento', 'indenizacao', 'sinistro', 'sinistralidade',
]
