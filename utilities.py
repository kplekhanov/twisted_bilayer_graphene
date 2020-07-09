import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, bmat

PRECISION = 4

def print_progress_bar(num, tot):
    num += 1
    l = 100 * num // tot
    num_str = "{0:.2f}%".format(l)
    tot_str = '#' * l + ' ' * (100 - l)
    tot_str = num_str + ' ' * (8 - len(num_str)) + '[' + tot_str + ']'
    print(tot_str, end='\r')

def get_inv(u1, u2):
    ## get reciprocal lattice vecs
    inv_matrix = np.linalg.inv(np.array((u1, u2)))
    return inv_matrix[:,0], inv_matrix[:,1]

def rotate_vec(vec, theta):
    return np.array(((np.cos(theta), -np.sin(theta)), (np.sin(theta), np.cos(theta)))) @ vec

def vec_to_tuple(vec, num_trunc=PRECISION):
    ## truncate vec (a numpy array, non-hashable) and convert it into a tuple (hashable)
    return tuple(np.around(vec, decimals=num_trunc))

def compose_hamdics(hamdic_arr):
    ## input should be of the type [[hamdic_00, hamdic_01, ...], [hamdic_10, hamdic_11, ...], ...]
    hamdic_keys = []
    dtype = np.float64
    for hamdic_list in hamdic_arr:
        for hamdic in hamdic_list:
            if hamdic != None:
                hamdic_keys += hamdic.dic.keys()
                if hamdic.dtype == np.complex128:
                    dtype = np.complex128
    hamdic_keys = set(hamdic_keys)
    
    hamdic_out = Hamdic(shape=(0,0), dtype=dtype)
    for key in hamdic_keys:
        spmat_arr = []
        for j1 in range(len(hamdic_arr)):
            hamdic_list = hamdic_arr[j1]
            spmat_list = []
            for j2 in range(len(hamdic_list)):
                hamdic = hamdic_list[j2]
                if hamdic is None:
                    spmat_list.append(None)
                elif key not in hamdic.dic.keys():
                    spmat_list.append(lil_matrix(hamdic.shape))
                else:
                    spmat_list.append(hamdic.dic[key])
            spmat_arr.append(spmat_list)
        hamdic_out.dic[key] = bmat(spmat_arr)
        if hamdic_out.shape == (0, 0): hamdic_out.shape = hamdic_out.dic[key].shape
        elif hamdic_out.dic[key].shape != hamdic_out.shape: raise Exception
        
    return hamdic_out

class Hamdic:
    def __init__(self, shape=(0,0), dtype=np.float64):
        self.dtype = dtype
        self.dic = {} ## a thingy of the type {key1:lil_matrix1, key2:lil_matrix2, ...}
        self.shape = shape
        
    def append(self, key, data_input):
        row, col, val = data_input
        if not key in self.dic.keys():
            self.dic[key] = lil_matrix(self.shape, dtype=self.dtype)
        self.dic[key][row,col] = val
                
    def __add__(self, hamdic_other):
        if self.shape != hamdic_other.shape:
            raise Exception
        if self.dtype == np.complex128 or hamdic_other == np.complex128:
            dtype = np.complex128
        else:
            dtype = np.float64
        hamdic_out = Hamdic(self.shape, dtype)
        for key in set(list(self.dic.keys()) + list(hamdic_other.dic.keys())):
            hamdic_out.dic[key] = lil_matrix(self.shape, dtype=dtype)
        for key in self.dic.keys():
            hamdic_out.dic[key] += self.dic[key]
        for key in hamdic_other.dic.keys():
            hamdic_out.dic[key] += hamdic_other.dic[key]
        return hamdic_out
    
    def get_hermitian_conjugate(self):
        hamdic = Hamdic((self.shape[1], self.shape[0]), self.dtype)
        for key in self.dic.keys():
            key_new = (-key[0], -key[1])
            hamdic.dic[key_new] = self.dic[key].conjugate().T
        return hamdic

class Cell:
    def __init__(self, u1, u2):
        ## defined such that the center of the unit cell has coordinates (0,0)
        self.u1, self.u2 = u1, u2
        self.u_origin = -(u1 + u2) / 2
        self.q1, self.q2 = get_inv(u1, u2)
        
    def rotate(self, theta):
        cell = Cell(self.u1, self.u2)
        cell.u1, cell.u2 = rotate_vec(cell.u1, theta), rotate_vec(cell.u2, theta)
        cell.q1, cell.q2 = rotate_vec(cell.q1, theta), rotate_vec(cell.q2, theta)
        cell.u_origin = rotate_vec(cell.u_origin, theta)
        return cell
        
    def check_vec(self, vec, epsilon=10**(-PRECISION)):
        ## check if vec is inside the unit cell
        return (-.5 < vec.dot(self.q1) + epsilon < .5) and (-.5 < vec.dot(self.q2) + epsilon < .5)
    
    def get_shift(self, vec, epsilon=10**(-PRECISION)):
        ## return the shift of vec with repsect to the boundaries of the unit cell
        return (vec.dot(self.q1) + .5 + epsilon) // 1, (vec.dot(self.q2) + .5 + epsilon) // 1
    
    def get_u_shift(self, shift_1, shift_2):
        return shift_1 * self.u1 + shift_2 * self.u2
    
    def draw(self, fax):
        vecs = [np.zeros(2), self.u1, self.u1+self.u2, self.u2, np.zeros(2)]
        vecs = [vec + self.u_origin for vec in vecs]
        fax.plot([vec[0] for vec in vecs], [vec[1] for vec in vecs])
        
    def draw_q(self, fax):
        vecs = [np.zeros(2), self.q1, self.q1+self.q2, self.q2, np.zeros(2)]
        fax.plot([vec[0] for vec in vecs], [vec[1] for vec in vecs])
        
class Lattice:
    def __init__(self, cell):
        self.cell = cell
        self.sites_int, self.sites_ext = [], []
        self.lookup = {}
        self.num_sites = 0
        
    def __add__(self, lattice_other):
        ## be warry -- non commutative!
        if self.cell != lattice_other.cell:
            raise Exception
        lattice_out = Lattice(self.cell)
        lattice_out.num_sites = self.num_sites + lattice_other.num_sites
        lattice_out.sites_int = self.sites_int + lattice_other.sites_int
        lattice_out.sites_ext = self.sites_ext + lattice_other.sites_ext
        lattice_out.lookup = self.lookup
        for key, val in lattice_other.lookup.items():
            lattice_out.lookup[key] = val + self.num_sites
        return lattice_out
        
    def add_site(self, vec):
        if self.cell.check_vec(vec):
            self.sites_int.append(vec)
            self.lookup[vec_to_tuple(vec)] = self.num_sites
            self.num_sites += 1
        else:
            self.sites_ext.append(vec)
            
    def get_hamdic_vec(self, lattice_other, vec, val, dtype=np.float64):
        if self.cell != lattice_other.cell: raise Exception
        hamdic = Hamdic((self.num_sites, lattice_other.num_sites), dtype)
        for i_site in range(self.num_sites):
            site_new = self.sites_int[i_site] + vec
            shift_1, shift_2 = self.cell.get_shift(site_new)
            try:
                vec_shifted = site_new - (shift_1 * self.cell.u1 + shift_2 * self.cell.u2)
                i_site_new = lattice_other.lookup[vec_to_tuple(vec_shifted)]
            except KeyError:
                continue
            hamdic.append((shift_1, shift_2), (i_site, i_site_new, val))
        return hamdic
    
    def get_hamdic_cutoff(self, lattice_other, val_ref, distance_cutoff_sq=1):
        ## hardcoded to be real only
        if self.cell != lattice_other.cell: raise Exception
        distances_sq_np = np.zeros((self.num_sites, lattice_other.num_sites))
        for j1 in range(self.num_sites):
            for j2 in range(lattice_other.num_sites):
                vec_diff = lattice_other.sites_int[j2] - self.sites_int[j1]
                shift_1, shift_2 = self.cell.get_shift(vec_diff)
                vec_diff -= shift_1 * self.cell.u1 + shift_2 * self.cell.u2
                distances_sq_np[j1,j2] = vec_diff[0] ** 2 + vec_diff[1] ** 2
        hamdic = Hamdic((self.num_sites, lattice_other.num_sites))
        for index in np.argsort(distances_sq_np, axis=None):
            j1, j2 = index // self.num_sites, index % self.num_sites
            distance_sq = distances_sq_np[j1,j2]
            if distance_sq > distance_cutoff_sq: break
            vec_1, vec_2 = self.sites_int[j1], lattice_other.sites_int[j2]
            shift_1, shift_2 = self.cell.get_shift(vec_2-vec_1)
            val = val_ref * np.exp(-np.sqrt(distance_sq/distance_cutoff_sq))
            hamdic.append((shift_1, shift_2), (j1, j2, val))
        return hamdic
            
    def draw(self, fax, color_int="black", color_ext="gray", u_offset=np.zeros(2), draw_numbers=False):
        for i in range(2):
            sites = (self.sites_int, self.sites_ext)[i]
            c = (color_int, color_ext)[i]
            x_arr = [site[0]+u_offset[0] for site in sites]
            y_arr = [site[1]+u_offset[1] for site in sites]
            fax.scatter(x_arr, y_arr, c=c, s=3.0, rasterized=True)
        if draw_numbers:
            for j in range(self.num_sites):
                x, y = self.sites_int[j]
                fax.text(x, y, "{}".format(j))