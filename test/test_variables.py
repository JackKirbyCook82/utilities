import os.path
from specs import specs_fromfile, specs_tofile

from variables import Geography, Datetime, create_customvariable

repository = os.path.dirname(os.path.realpath(__file__))
specs_file = os.path.join(repository, 'myspecs.csv')

specs = specs_fromfile(specs_file, splitchar=';')
specs_tofile(specs_file, specs, concatchar=';')


geography = Geography(**dict(state={'value':48, 'name':'TX'}, county={'value':157, 'name':'FortBend'}, subdivision={'value':'*'}))
print(geography.name)
print(geography.variabletype)
print(str(geography))
print(repr(geography))
print(geography.geoid)
print(Geography.fromstr(str(geography))) 
print('\n')


geography = Geography(**dict(state=48, county=157, subdivision='*'))
print(geography.name)
print(geography.variabletype)
print(str(geography))
print(geography.geoid)
print(repr(geography))
print(Geography.fromstr(str(geography)))
print('\n')


datetime = Datetime(year=2017, month=2, day=15)
print(datetime.name)
print(datetime.variabletype)
print(str(datetime))
print(repr(datetime))
print(Datetime.fromstr(str(datetime)))
print('\n')


Households = create_customvariable(specs['households'])
households = Households(50550000)
print(households.name)
print(households.variabletype)
print(households.spec.data)
print(str(households))
print(repr(households))
print(Households.fromstr(str(households)))
print(Households(10110000) + Households(20220000))
print('\n')


Population = create_customvariable(specs['population'])
population = Population(101110000)
print(population.name)
print(population.variabletype)
print(population.spec.data)
print(str(population))
print(repr(population))
print(Population.fromstr(str(population)))
print('\n')


Income = create_customvariable(specs['income'])
income = Income([25000, 50000])
print(income.name)
print(income.variabletype)
print(income.spec.data)
print(str(income))
print(repr(income))
print(Income.fromstr(str(income)))
print(Income([25000, 50000]) + Income([50000, None]))
print(Income([25000, 50000]) - Income([35000, 50000]))
print('\n')


ratio = population / households
print(ratio.name)
print(ratio.variabletype)
print(ratio.spec.data)
print(str(ratio))
print(repr(ratio))
print(type(ratio).fromstr(str(ratio)))















