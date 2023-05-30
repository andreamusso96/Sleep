import pandas as pd

from AdminDataProcessing.Data import Population, Equipements


class BusinessDensityIndex:
    def __init__(self, equipements: Equipements, population: Population):
        self.equipements = equipements
        self.population = population

    def compute_index(self):
        number_of_businesses_by_iris = self.equipements.data.groupby('iris')['NB_EQUIP'].sum()
        population_in_iris = self.population.data.set_index('iris')['P19_POP'].astype(int)
        number_of_businesses_and_population = pd.merge(number_of_businesses_by_iris, population_in_iris, left_index=True, right_index=True)
        business_density_index = (number_of_businesses_and_population['NB_EQUIP'] / number_of_businesses_and_population['P19_POP']).to_frame().clip(lower=0, upper=1).rename(columns={0: 'business_density_index'})
        return business_density_index


if __name__ == '__main__':
    equipements = Equipements()
    population = Population()
    equipements.load()
    population.load()
    business_density_index = BusinessDensityIndex(equipements, population)
    bi = business_density_index.compute_index()
    print(bi.head())


