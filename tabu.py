import sys
import matplotlib.pyplot as plt
from
solution
import
Solution


def tabu_search(max_iter, tabu_tenure, neighbor_size=1000, init='No'):
 tabu = dict()
 sol = Solution()
 if init == 'NN':
 sol.nearest_neighborhood_initialization()
 for i in range(sol.number_of_nodes):
 for j in range(sol.number_of_nodes):
 tabu[(i, j)] = 0
 obj = []
 count = 0
 best_obj = sys.float_info.max
 while count <= max_iter:
 pair = sol.best_neighbor_w_tabu_aspiration(neighbor_size, tabu,
best_obj)
 tabu[pair] += tabu_tenure
 for i in range(sol.number_of_nodes):
 for j in range(sol.number_of_nodes):
 if tabu[(i, j)] > 0:
 tabu[(i, j)] -= 1
 sol.swap_operation(pair[0], pair[1])
 obj.append(sol.get_obj_func_value())
 count += 1
 if sol.get_obj_func_value() < best_obj:
 best_obj = sol.get_obj_func_value()
 print('incumbent value: ', str(best_ob
plt.plot(list(range(len(obj))), obj)
 plt.xlabel('Iteration No')
 plt.ylabel('Objective Function Value')
 plt.title('Tabu Search')
 plt.show()
if __name__ == '__main__':
 import time
 s=time.time()
 tabu_search(1000, 100, init='NN')
 e=time.time()
 print('cpu time: ', str(e-s))
import
pandas
as pd
def read_tsp_data():
 """
 This function reads the STSP data by using
 pandas and returns the data Numpy array
 format.
 :return: 2D Numpy Array
 """
 return pd.read_csv('INDR568_HW2_STSP_instance.csv').values[:,
1:]
from read
import
read_tsp_data
import random
import copy
import sys
class Solution(object):
 def __init__(self):
 random.seed(521)
 self.distance_matrix=read_tsp_data()  # read the distance matrix
 self.number_of_nodes=self.distance_matrix.shape[0]  # extract the number of
nodes in the system
 """
 Give a random initial solution by shuffling
 """
 self.route=list(range(self.number_of_nodes))
 random.shuffle(self.route)
def get_obj_func_value(self):
 """
 This function calculates and returns the objective function value
 :return: float
 """
 obj_value=0
 for i in range(self.number_of_nodes-1):
 obj_value += self.distance_matrix[self.route[i], self.route[i + 1]]
 return obj_value
def swap_operation(self, ind1, ind2):
 """
 This function carries out the swap operation
 between given two indices if they are valid.
 :param ind1: int
 :param ind2: int
 :return: None, **INPLACE**
 """
 if ind1 >= self.number_of_nodes or ind2 >= self.number_of_nodes:
 raise ValueError('Your indices must be in the appropriate range')
 else:
 node1=self.route[ind1]
 node2=self.route[ind2]
 self.route[ind1]=node2
 self.route[ind2]=node1
def get_route(self):
 return self.route
 def set_route(self, route):
 self.route=route
 def nearest_neighborhood_initialization(self):
 ls=[0]
while len(ls) < self.number_of_nodes:
 current=ls[-1]
 candidates=dict()
 for i in range(self.number_of_nodes):
 if i not in ls:
 candidates[i]=self.distance_matrix[current, i]
 ls.append(min(candidates.items(), key=lambda x: x[1])[0])
 self.set_route(ls)
def define_neighbors(self, neighbor_num):
 pairs=[]
 count=0
 while count < neighbor_num:
 ind1=self.route.index(random.choice(self.route))
 ind2=self.route.index(random.choice(self.route))
 if ind1 == ind2:
 ind1=self.route.index(random.choice(self.route))
 ind2=self.route.index(random.choice(self.route))
 if ind1 == ind2:
 ind1=self.route.index(random.choice(self.route))
 ind2=self.route.index(random.choice(self.route))
 pairs.append((ind1, ind2))
 count += 1
 return pairs
def best_neighbor(self, neighbor_num):
 neighbors=self.define_neighbors(neighbor_num)
 which_pair=0
 tmp=sys.float_info.max
 for pair in neighbors:
 candidate_sol=copy.deepcopy(self)
 candidate_sol.swap_operation(pair[0], pair[1])
 if candidate_sol.get_obj_func_value() < tmp:
 which_pair=pair
 tmp=candidate_sol.get_obj_func_value()
 print(which_pair)
 print(tmp)
 return which_pair
def best_neighbor_w_tabu(self, neighbor_num, tabu):
 """
 :param neighbor_num: int
 :param tabu_tenure: dictionary, keys = pairs, values = how much left
 :return: which pair, (int, int)
 """
 neighbors=self.define_neighbors(neighbor_num)
 which_pair=0
 tmp=sys.float_info.max
 for pair in neighbors:
 if tabu[pair] == 0:
 candidate_sol=copy.deepcopy(self)
 candidate_sol.swap_operation(pair[0], pair[1])
 if candidate_sol.get_obj_func_value() < tmp:
which_pair=pair
 tmp=candidate_sol.get_obj_func_value()
 print(which_pair)
 print(tmp)
 return which_pair
def best_neighbor_w_tabu_aspiration(self, neighbor_num, tabu, best_obj):
 """
 :param neighbor_num: int
 :param tabu_tenure: dictionary, keys = pairs, values = how much left
 :return: which pair, (int, int)
 """
 neighbors=self.define_neighbors(neighbor_num)
 our_dict={}
 for i in neighbors:
 dummy_sol=copy.deepcopy(self)
 dummy_sol.swap_operation(i[0], i[1])
 our_dict[i]=dummy_sol.get_obj_func_value()
 updated_dict={}
 updated_dict[w]=our_dict[w]
 if min(updated_dict.items(), key=lambda x: x[1])[1] < best_obj:
 return min(updated_dict.items(), key=lambda x: x[1])[0]
 else:
 which_pair=0
 tmp=sys.float_info.max
 for w in sorted(updated_dict, key=updated_dict.get):
 if tabu[w] == 0:
 candidate_sol=copy.deepcopy(self)
 candidate_sol.swap_operation(w[0], w[1])
 if candidate_sol.get_obj_func_value() < tmp:
 which_pair=w
 tmp=candidate_sol.get_obj_func_value()
 print(which_pair)
 print(tmp)
 return which_pair
