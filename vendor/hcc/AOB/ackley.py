from AOB.Benchmarks import Benchmarks
import numpy as np
import yaml


class ackley(Benchmarks):
    def __init__(self,ID, output_path, data_dir=None):
        super().__init__(output_path, data_dir=data_dir)
        self.ID = ID
        info_file_path = self.data_dir / f'F{ID}-info.txt'

        with open(info_file_path, "r") as file:
            data = yaml.safe_load(file)

        self.s_size = data['sub_num']
        self.dimension = data['dimension']
        self.dimension_real = data.get('dimension_real', self.dimension)
        self.overlap = data['overlap_degree']

        self.Ovector = self.readOvector()
        self.Pvector = self.readPermVector()


        self.s = self.readS(self.s_size)
        self.w = self.readW(self.s_size)

        self.minX = data['lower_bound']
        self.maxX = data['upper_bound']


        self.anotherz = np.zeros(self.dimension)
        self.cache_Rotation = {i: self.readR(i) for i in data['subgroups_type']}


    def __call__(self, x):
        return self.compute(x)

    def info(self):
        info = {'best': 0.0, 'dimension': self.dimension, 'lower': self.minX, 'threshold': 0, 'upper': self.maxX}
        return info

    def compute(self, x):

        x = self.prepare_input(x)
        
        result = np.zeros(x.shape[0])

        c = 0
        self.anotherz = x - self.Ovector  # Element-wise subtraction

        for i in range(self.s_size):
            anotherz1 = self.rotateVectorConform(i, c)
            anotherz1 = self.transform_osz(anotherz1)
            anotherz1 = self.transform_asy(anotherz1, 0.2)
            result += self.w[i] * self.ackley(anotherz1)
            c += self.s[i]  

        self.fitness_record.extend(result.tolist())
        
        return result
