import geopandas as gpd
import matplotlib.pyplot as plt

import Data.plotting as plotting
import Data.election_france as ef

if __name__ == '__main__':
    iris =['751010101', '751010102', '751010103', '751010104', '751010105',
       '751010199', '751010201', '751010202', '751010203', '751010204',
       '751010205', '751010206', '751010301', '751010302', '751010303',
       '751010401', '751010402', '751020501', '751020502', '751020503']
    ef.get_percentage_votes_for_party_at_iris_level(list_number=[1, 2, 3, 7, 9, 10, 12, 15, 17, 18, 19, 20, 21, 23, 29, 30], iris=iris)