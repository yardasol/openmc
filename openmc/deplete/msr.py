from collections import OrderedDict

from openmc import Material

class MsrContinuous:

    def __init__(self, local_mats):
        self.index_burn = [mat.id for mat in local_mats if mat.depletable]
        self.ord_burn = OrderedDict((int(id), i) for i, id in enumerate(self.index_burn))
        self.removal_terms = OrderedDict((id, OrderedDict()) for id in self.index_burn)

    def _get_mat_index(self, mat):
        """Helper method for getting material index"""
        if isinstance(mat, Material):
            mat = str(mat.id)
        return self.index_mat[mat] if isinstance(mat, str) else mat

    @property
    def n_burn(self):
        return len(self.index_burn)

    def index_msr(self):

        index_msr = OrderedDict(((i, i), None) for i in range(self.n_burn))
        for id, val in self.removal_terms.items():
            if val:
                for elm, [tr, mat] in val.items():
                    if mat is not None:
                        j = self.ord_burn[id]
                        i = self.ord_burn[mat.id]
                        index_msr[(i,j)] = None

        return index_msr.keys()

    def add_removal_rate(self, mat, elements, removal_rate, units = '1/s', dest_mat=None):

        if not isinstance(elements, list):
            raise ValueError('Elements must be a list')
        else:
            if not all(isinstance(element, str) for element in elements):
                raise ValueError('Each element must be a str')

        if not isinstance(removal_rate, float):
            raise ValueError('Removal rate must be a float')

        if not isinstance(mat, Material):
            raise ValueError(f'{mat} is not a valid openmc material')
        else:
            if not mat.depletable:
                raise ValueError(f'{mat} must be depletable')

        if dest_mat is not None:
            if not isinstance(dest_mat, Material):
                raise ValueError(f'{dest_mat} is not a valid openmc material')
            else:
                if not dest_mat.depletable:
                    raise ValueError(f'{dest_mat} must be depletable')

        for element in elements:
            self.removal_terms[mat.id][element] = [removal_rate, dest_mat]

    def get_removal_rate(self, mat, element):
        if isinstance(mat, Material):
            if mat.depletable:
                mat = mat.id
            else:
                raise ValueError(f'{mat} must be depletable')
        else:
            if mat not in self.index_burn:
                raise ValueError(f'{mat} is not a valid depletable material id')

        return self.removal_terms[mat][element][0]

    def get_destination_mat(self, mat, element):

        return self.removal_terms[mat.id][element][1]

    def get_elements(self, mat):
        elements=[]
        for k, v in self.removal_terms.items():
            if k == mat:
                for elm, _ in v.items():
                    elements.append(elm)
        return elements
