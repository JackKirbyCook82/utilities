import os.path
import tables as tbls
from tables.transformations import Normalize, Cumulate
from parsers import ListParser, DictParser, FormatParser, FormatValueParser
from parsers.valueformatters import ValueFormatter
from parsers.valuegenerators import NumGenerator, ValueGenerator
from uscensus import USCensus_WebAPI
from variables import Variables
from utilities.arrays import xarray_fromdata

tbls.set_options(linewidth=100, maxrows=30, maxcolumns=10, threshold=100, precision=3)

apikey = 'f98e5cb368f964cde784b85a0b22035efc3a3498'
downloadrate = 3
headers = {'USER_AGENT' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.121 Safari/537.36'}

repository = os.path.dirname(os.path.realpath(__file__))
tables_file = os.path.join(repository, 'myuscensustables.csv')
variables_file = os.path.join(repository, 'myvariables.csv')

listparser = ListParser(';')
dictparser = DictParser(pattern=';=', emptyvalues=False)
tagsparser = FormatParser(NumGenerator(*'&-/'), pattern=';|&=')    
conceptsparser = FormatValueParser(ValueGenerator.createall(*'&-/'), ValueFormatter.createall(delimiter='-'), pattern=';|&=') 

variables = Variables.fromfile(variables_file, splitchar=';')

normalize = Normalize()
cumulate = Cumulate(direction='upper')

tables_parsers = dict(parms=listparser, tags=tagsparser, preds=dictparser, concepts=conceptsparser)
webapi = USCensus_WebAPI(apikey, tables_file=tables_file, tables_parsers=tables_parsers, repository=repository, headers=headers, downloadrate=downloadrate)
webapi.setitems(universe='households', index='geography', headers='income')
webapi['tenure'] = 'Owner'

values = ['$100K US - $200K US', '$200K US - $250K US', '$250K US - $500K US']
geography = variables['geography'](**{'state':48, 'county':157, 'subdivision':'*'})
dates = variables['date'](year=2017)   
estimate = 5       

print(repr(webapi), '\n')
print(str(webapi), '\n')

tables = []
for value in values:
    webapi['value'] = value
    dataframe = webapi(geography=geography, dates=dates, estimate=estimate)
    xarray = xarray_fromdata('dataframe', dataframe, 'households')
    table = tbls.ArrayTable(xarray, variables=variables, name='USCensus Test')
    tables.append(table)
    
table = tables[0].combine(tables[1], onscope='value')
table = table.append(tables[2], toaxis='value')
table = normalize(table, axis='income')
table = cumulate(table, axis='income')
print(str(table))

