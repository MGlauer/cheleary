from pysmiles.read_smiles import _tokenize, TokenType
import re
import numpy as np
import tensorflow as tf

class Encoder:
    @property
    def shape(self):
        raise NotImplementedError

    def run(self, input):
        raise NotImplementedError


class SmilesOneHotEncoder(Encoder):
    # atom_chars = sorted(['C@@H', 'C@H', 'N@H+', 'Nb', 'Ta', 'N', 'c', 'n', 'CH', 'O', 'C', 'P', 'S', 'Cl', 'nH', 's', 'Br', 'o', 'I', 'H', '*', 'F', 'Ca', 'Al', 'OH', 'Na', 'NH', 'Se', 'Co', 'Hg', 'As', 'Mg', 'Cu', 'Si', 'Au', 'Tc', 'B', 'Fe', 'Ge', 'Sm', 'Ru', 'V', 'Mo', 'He', 'Sb', 'Yb', 'Gd', 'Li', 'Cr', 'Ag', 'Fr', 'Ba', 'Pb', 'Y', 'Sr', 'Ga', 'Eu', 'Mn', 'Os', 'Tl', 'In', 'Sn', 'Ir', 'La', 'Lu', 'Cs', 'Ce', 'W', 'Zn', 'Be', 'Bi', 'U', 'Ni', 'Ho', 'Pt', 'Rb', 'K', 'SeH', 'TeH', 'Te', 'At', 'Re', 'Ra', 'Ti', 'SiH', 'se', 'pH', 'te', 'Ar', 'Xe', 'Kr', 'Cd', 'Pd', 'Rh', 'cH', 'p', 'Ne', 'Rn', 'LiH', 'Zr', 'AsH', 'Pr', 'Po', 'Tb'], key=lambda x: -len(x))
    def __init__(self):
        self.__atom_chars = ["\*", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl",
                      "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se",
                      "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb",
                      "Te", "I", "Xe", "Cs", "Ba", "La", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb",
                      "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh",
                      "Fl", "Mc", "Lv", "Ts", "Og", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm",
                      "Yb", "Lu", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "c",
                      "n", "s", "o", "se", "p"]

        assert len(self.__atom_chars) == 118 + 7, len(self.__atom_chars)
        self.__modifier_chars = ['+', '++', '-', '--', '@', '@@', '@+', '@@+', ":"]
        self.__bond_chars = [".", "-", "=", "#", "$", "/", "\\"]
        self.__branch_chars = ["(", ")"]
        self.__mults = 12
        self.__mult_regex = "|".join(map(str, range(1, self.__mults + 1)))
        self.__input_lenth = len(self.__atom_chars) + len(self.__modifier_chars) + 1 + 2 * self.__mults + len(self.__bond_chars) + 36 + len(self.__branch_chars)

        self.__atom_regex = rf"({'|'.join(self.__atom_chars)})"
        self.__complex_atom_regex = fr"(?P<lhr>{self.__atom_regex}+)(?P<atom_modifier>{'|'.join(map(re.escape, self.__modifier_chars))})(?P<rhr>{self.__atom_regex}+)"

    @property
    def shape(self):
        raise (None, self.__input_lenth)

    def run(self, input):
        return [self._encode_token(smiles) for smiles in _tokenize(input)]

    def _encode_token(self, token):
        v = np.zeros(self.__input_lenth, dtype=np.bool)
        t, x = token
        indices = []
        offset = 0
        if t == TokenType.ATOM:
            regex = re.compile(rf"^(?P<isotopes>\d+)?"
                               rf"((?P<atoms>({self.__atom_regex})*)|(?P<complex_atom>({self.__complex_atom_regex})+))"
                               rf"(?P<multiplicity_0>{self.__mult_regex})?"
                               rf"(?P<modifier>{'|'.join(map(re.escape, self.__modifier_chars))})?"
                               rf"(?P<multiplicity_1>{self.__mult_regex})?$")
            for y in ["[", "]"]:
                x = x.replace(y, "")
            match = regex.match(x)
            if match is not None:
                offset = 0
                if match.group("atoms"):
                    self._encode_atoms(match.group("atoms"), indices)
                if match.group("complex_atom"):
                    self._encode_atoms(match.group("lhr"), indices)
                    indices.append(self.__modifier_chars.index(match.group("atom_modifier")) + offset)
                    self._encode_atoms(match.group("rhr"), indices)
                offset += len(self.__atom_chars)
                if match.group("modifier"):
                    indices.append(self.__modifier_chars.index(match.group("modifier")) + offset)
                offset += len(self.__modifier_chars)
                if match.group("isotopes"):
                    indices.append(offset)
                offset += 1
                for i in range(0, 2):
                    if match.group("multiplicity_" + str(i)):
                        indices.append(int(match.group("multiplicity_" + str(i))) + offset + self.__mults * i)
            else:
                raise Exception("Could not encode atom", x)
        else:
            offset += len(self.__atom_chars) + len(self.__modifier_chars) + 1 + 2 * self.__mults
            if t == TokenType.BOND_TYPE or t == TokenType.EZSTEREO:
                indices.append(self.__bond_chars.index(x))

            else:
                offset += len(self.__bond_chars)
                if t == TokenType.RING_NUM:
                    indices.append(x + offset - 1)
                else:
                    offset += 36
                    if t == TokenType.BRANCH_START or t == TokenType.BRANCH_END:
                        indices.append(self.__branch_chars.index(x))
        if not indices:
            raise Exception("Could not encode", x)
        for index in indices:
            v[index] = True
        return v

    def _encode_atoms(self, group, indices):
        for submatch in re.finditer(self.__atom_regex, group):
            atom = submatch.groups()[0]
            if atom == "*":
                atom = "\*"
            indices.append(self.__atom_chars.index(atom))


class AtomOrdEncoder(Encoder):
    def __init__(self):
        __tokens = {1: ['*', '11B', '13C', '13C@', '13C@@', '13C@@H', '13C@H', '13CH', '13CH2', '13CH3', '13c', '13cH', '14c', '15N',
             '15N+', '15N-', '15NH', '15NH2', '15NH3', '15NH4', '15n', '15nH', '18F', '18O', '18O-', '18OH', '1CH', '1H',
             '208PbH2', '24O', '2H', '2H+', '2H-', '2NaH', '3H', '3NH4', '3NaH', '4NaH', '54Fe+3', '57Fe+3', '6Li+', '6Na+',
             'Ac+', 'Ag', 'Ag+', 'Ag++', 'Ag+3', 'Ag-', 'Al', 'Al+', 'Al++', 'Al+3', 'Al+4', 'Al+5', 'Al+6', 'Al+7', 'Al+8',
             'Al+9', 'AlH', 'AlH+', 'AlH++', 'AlH2', 'AlH2+', 'AlH3', 'AlH3++', 'AlH4+7', 'As', 'As+', 'Au', 'Au+', 'Au++',
             'Au+3', 'Au+4', 'Au+5', 'Au+7', 'Au-', 'AuH2+3', 'B', 'B+', 'B++', 'B+3', 'B-', 'B--', 'B-3', 'B@', 'B@@-',
             'BH', 'BH-', 'BH2-', 'BH3', 'BH3-', 'BH3--', 'BH4-', 'Ba', 'Ba+', 'Ba++', 'Ba+6', 'Ba+7', 'BaH+', 'BaH2',
             'Be++', 'Bi', 'Bi+', 'Bi++', 'Bi+10', 'Bi+3', 'Bi+4', 'Bi+5', 'Bi+6', 'BiH', 'BiH++', 'BiH+3', 'BiH2',
             'BiH2++', 'BiH2+3', 'BiH3', 'BiH3++', 'Br', 'Br+', 'Br-', 'Br--', 'Br-3', 'Br0', 'Br1', 'Br2', 'Br3', 'Br4',
             'Br5', 'Br6', 'Br7', 'Br8', 'Br9', 'BrH', 'BrH2--', 'C', 'C+', 'C++', 'C-', 'C--', 'C-3', 'C-4', 'C0', 'C1',
             'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C@', 'C@-', 'C@0', 'C@2', 'C@3', 'C@9', 'C@@', 'C@@-', 'C@@1',
             'C@@3', 'C@@5', 'C@@7', 'C@@H', 'C@@H0', 'C@@H1', 'C@@H2', 'C@@H4', 'C@@H6', 'C@H', 'C@H0', 'C@H1', 'C@H2',
             'C@H3', 'C@H4', 'CH', 'CH+', 'CH+3', 'CH-', 'CH0', 'CH1', 'CH2', 'CH2+', 'CH2-', 'CH20', 'CH21', 'CH22',
             'CH23', 'CH24', 'CH25', 'CH26', 'CH27', 'CH28', 'CH29', 'CH3', 'CH3-', 'CH30', 'CH31', 'CH32', 'CH33', 'CH34',
             'CH35', 'CH36', 'CH37', 'CH38', 'CH39', 'CH4', 'CH4-', 'CH6', 'CH7', 'Ca', 'Ca+', 'Ca++', 'Ca+4', 'Ca+6',
             'CaH', 'CaH+', 'CaH2', 'Cd', 'Cd+', 'Cd++', 'Ce', 'Ce+', 'Ce++', 'Ce+10', 'Ce+3', 'Ce+4', 'Ce+6', 'Ce+7',
             'Ce+8', 'Cl', 'Cl+', 'Cl-', 'Cl--', 'Cl-3', 'Cl0', 'Cl1', 'Cl2', 'Cl3', 'Cl4', 'Cl5', 'Cl6', 'Cl7', 'Cl8',
             'Cl9', 'ClH', 'ClH--', 'ClH0', 'ClH2', 'ClH3', 'ClH3--', 'ClH4', 'ClH5', 'ClH6', 'ClH7', 'ClH8', 'Co', 'Co+',
             'Co++', 'Co+3', 'Co+4', 'Co+5', 'Co+6', 'Co+7', 'Co+8', 'Co+9', 'Co--', 'Co-3', 'CoH5+8', 'CoH6+9', 'Cr',
             'Cr+', 'Cr++', 'Cr+3', 'Cr+4', 'Cr+5', 'Cr+6', 'Cr+7', 'Cr+9', 'Cr-3', 'Cs', 'Cs+', 'CsH', 'Cu', 'Cu+', 'Cu++',
             'Cu+3', 'Cu+4', 'Cu+5', 'Cu+6', 'Cu-', 'CuH2', 'CuH2++', 'CuH3+5', 'CuH4', 'CuH4+6', 'Dy', 'Dy++', 'Dy+3',
             'Er', 'Er+', 'Er++', 'Er+3', 'Eu', 'Eu+', 'Eu++', 'Eu+11', 'Eu+3', 'Eu+6', 'Eu+9', 'F', 'F+', 'F-', 'F--',
             'F0', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'FH+', 'Fe', 'Fe+', 'Fe++', 'Fe+3', 'Fe+4', 'Fe+5',
             'Fe+6', 'Fe+8', 'Fe+9', 'Fe-', 'Fe-3', 'Fe-4', 'FeH2', 'FeH3', 'FeH4+7', 'FeH5+8', 'FeH6+9', 'Ga', 'Ga++',
             'Ga+3', 'Ga+4', 'Ga+6', 'GaH2', 'GaH3', 'Gd', 'Gd+', 'Gd++', 'Gd+11', 'Gd+3', 'Ge', 'Ge++', 'Ge+3', 'Ge+4',
             'Ge+8', 'GeH', 'GeH2', 'GeH3', 'GeH4', 'H', 'H+', 'H-', 'H--', 'H-3', 'H0', 'H1', 'H2', 'H3', 'H4', 'H6', 'H7',
             'H8', 'HH--', 'HeH', 'Hf', 'Hf+', 'Hf++', 'Hf+12', 'Hf+3', 'Hf+4', 'Hf+6', 'Ho', 'Ho+', 'Ho++', 'Ho+3', 'I',
             'I+', 'I-', 'I--', 'I-3', 'I0', 'I1', 'I2', 'I7', 'IH+', 'IH2+', 'In', 'In+', 'In+3', 'InH3', 'Ir', 'Ir+',
             'Ir++', 'Ir+10', 'Ir+3', 'Ir+4', 'Ir+5', 'Ir+6', 'Ir+7', 'IrH4+7', 'K', 'K+', 'K++', 'K+3', 'KH', 'KH2', 'La',
             'La+', 'La++', 'La+3', 'La+5', 'La+6', 'Li', 'Li+', 'LiH', 'Lu', 'Lu++', 'Lu+3', 'Lu+6', 'Mg', 'Mg+', 'Mg++',
             'Mg+4', 'Mg+6', 'Mg+7', 'Mg+8', 'MgH', 'MgH+', 'MgH2', 'MgH4+6', 'Mn', 'Mn+', 'Mn++', 'Mn+3', 'Mn+4', 'Mn+5',
             'Mn+6', 'Mn+7', 'Mn+8', 'Mn+9', 'MnH4+7', 'MnH5+8', 'Mo', 'Mo+', 'Mo++', 'Mo+3', 'Mo+4', 'Mo+5', 'Mo+6',
             'Mo+7', 'Mo+8', 'N', 'N+', 'N++', 'N+0', 'N+1', 'N+2', 'N+5', 'N+9', 'N-', 'N--', 'N-3', 'N0', 'N1', 'N2',
             'N3', 'N4', 'N5', 'N6', 'N7', 'N8', 'N9', 'N@', 'N@+', 'N@@', 'N@@+', 'NH', 'NH+', 'NH-', 'NH--', 'NH0', 'NH1',
             'NH2', 'NH2+', 'NH2-', 'NH20', 'NH21', 'NH22', 'NH23', 'NH24', 'NH25', 'NH26', 'NH27', 'NH28', 'NH29', 'NH3',
             'NH3+', 'NH3-', 'NH4', 'NH4+', 'NH4-', 'NH5', 'NH6', 'NH7', 'Na', 'Na+', 'Na++', 'Na+0', 'Na+1', 'Na+5', 'NaH',
             'NaH2', 'Nb', 'Nb++', 'Nb+3', 'Nb+4', 'Nb+5', 'Nd', 'Nd++', 'Nd+3', 'Nd+6', 'Ni', 'Ni+', 'Ni++', 'Ni+3',
             'Ni+4', 'Ni+5', 'Ni+6', 'Ni+8', 'Ni--', 'NiH2', 'O', 'O+', 'O-', 'O--', 'O-0', 'O-1', 'O-2', 'O-3', 'O-4',
             'O-5', 'O-9', 'O0', 'O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7', 'O8', 'O9', 'OH', 'OH+', 'OH-', 'OH--', 'OH0',
             'OH1', 'OH2', 'OH2+', 'OH2-', 'OH2--', 'OH2-3', 'OH2-4', 'OH3', 'OH3-', 'OH3--', 'OH3-3', 'OH3-5', 'OH4',
             'OH4-4', 'OH5', 'OH6', 'OH7', 'OH8', 'OH9', 'Os', 'Os+3', 'Os+4', 'Os+5', 'Os+6', 'Os+8', 'P', 'P+', 'P+5',
             'P-', 'P--', 'P-19', 'P-3', 'P@', 'P@-', 'P@@', 'P@@-', 'PH', 'PH+', 'PH-', 'PH2', 'PH2-', 'PH3+', 'Pb',
             'Pb++', 'Pb+3', 'Pb+4', 'PbH', 'PbH+', 'PbH+3', 'PbH2', 'PbH4', 'Pd', 'Pd+', 'Pd++', 'Pd+3', 'Pd+4', 'Pd+5',
             'Pd+6', 'Pd+7', 'Pd-', 'Pd--', 'Pm+3', 'Pr', 'Pr+3', 'Pr+4', 'Pr+9', 'Pt', 'Pt+', 'Pt++', 'Pt+10', 'Pt+3',
             'Pt+4', 'Pt+6', 'Pt--', 'PtH4+6', 'PtH6+10', 'Rb+', 'RbH', 'Re', 'Re+', 'Re+12', 'Re+3', 'Re+4', 'Re+5',
             'Re+6', 'Re+7', 'Rh', 'Rh+', 'Rh++', 'Rh+3', 'Rh+4', 'Rh+5', 'Rh+6', 'Rh+9', 'Rh-', 'Rh--', 'RhH', 'Ru', 'Ru+',
             'Ru++', 'Ru+3', 'Ru+4', 'Ru+5', 'Ru+6', 'Ru+7', 'Ru+8', 'Ru+9', 'Ru-3', 'Ru-4', 'RuH+3', 'RuH2+4', 'RuH6+8',
             'S', 'S+', 'S+6', 'S-', 'S--', 'S0', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S@', 'S@+', 'S@@',
             'S@@+', 'SH', 'SH+', 'SH-', 'SH--', 'SH2', 'SH2--', 'Sb', 'Sb++', 'Sb+11', 'Sb+3', 'Sb+4', 'Sb+5', 'Sb+7',
             'Sb-3', 'SbH', 'SbH+', 'SbH+4', 'SbH+5', 'SbH2', 'SbH2+3', 'SbH3', 'Sc', 'Sc++', 'Sc+3', 'Sc+6', 'Se', 'Se+',
             'Se-', 'Se--', 'SeH', 'SeH-', 'SeH2', 'Si', 'Si+', 'Si+4', 'Si-', 'Si--', 'SiH', 'SiH2', 'SiH3', 'SiH3-',
             'SiH4', 'Sm', 'Sm++', 'Sm+3', 'Sn', 'Sn+', 'Sn++', 'Sn+10', 'Sn+3', 'Sn+4', 'Sn+6', 'SnH', 'SnH+3', 'SnH2',
             'SnH3', 'SnH4', 'SnH6+10', 'Sr', 'Sr++', 'Sr+4', 'SrH2', 'Ta', 'Ta++', 'Ta+3', 'Ta+4', 'Ta+5', 'Tb', 'Tb++',
             'Tb+3', 'Te', 'Te++', 'Te+4', 'Te--', 'TeH', 'TeH-', 'TeH2', 'Th+4', 'Ti', 'Ti+', 'Ti++', 'Ti+3', 'Ti+4',
             'Ti+5', 'Ti+6', 'Ti+7', 'TiH4+6', 'Tl', 'Tl+', 'Tl+3', 'TlH', 'TlH+', 'Tm', 'Tm+3', 'U+', 'V', 'V+', 'V++',
             'V+10', 'V+3', 'V+4', 'V+5', 'V+6', 'V+7', 'V+8', 'VH2+3', 'VH7+8', 'VH8+10', 'W', 'W+', 'W++', 'W+10', 'W+3',
             'W+4', 'W+5', 'W+6', 'WH7+10', 'Xe', 'Y', 'Y+', 'Y++', 'Y+3', 'Y+6', 'Yb', 'Yb++', 'Yb+3', 'Yb+6', 'Zn', 'Zn+',
             'Zn++', 'Zn+3', 'Zn+4', 'Zn+5', 'Zn+6', 'Zn--', 'ZnH2', 'ZnH4+6', 'Zr', 'Zr+', 'Zr++', 'Zr+3', 'Zr+4', 'Zr+5',
             'Zr+6', 'c', 'c+', 'c-', 'c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'cH', 'cH+', 'cH-', 'cH0',
             'cH1', 'cH2', 'cH3', 'cH4', 'cH5', 'cH6', 'cH7', 'cH8', 'cH9', 'n', 'n+', 'n-', 'n--', 'n0', 'n1', 'n2', 'n3',
             'n4', 'n5', 'n6', 'n7', 'nH', 'nH+', 'nH-', 'o', 'o+', 'o0', 'o1', 'o2', 'o3', 'o6', 'p', 'pH', 's', 's+',
             's-', 's0', 's1', 's2', 's3', 's4', 's6', 's7', 's9', 'se', 'se+'], 2: ['#', '-', '.', '='],
         5: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25], 6: ['/', '\\'],
         4: [')'], 3: ['(']}

        offsets = {t: int(sum(len(__tokens[t2]) for t2 in range(1, t)) * 1.05) for t in __tokens}

        self.lookup = {t: i+offsets[key] for key, values in __tokens.items() for (i, t) in enumerate(values)}

    @property
    def shape(self):
        raise (None, 1)

    def _encode_token(self, token):
        t, x = token
        if isinstance(x, str):
            x = re.sub(r":\d", "", x.replace("[", "").replace("]", "")) # brush up token a bit
        return self.lookup[x]

    def run(self, input):
        return [self._encode_token(smiles) for smiles in _tokenize(input)]


class CharacterOrdEncoder(Encoder):

    @property
    def shape(self):
        raise (None, 1)

    def run(self, input):
        return [ord(c) for c in input]


class IntEncoder(Encoder):

    @property
    def shape(self):
        raise (None, 1)

    def run(self, input):
        return [int(c) for c in input]



