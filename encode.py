from pysmiles.read_smiles import _tokenize, TokenType
import re
import numpy as np

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


class SmilesAtomEncoder(Encoder):
    # atom_chars = sorted(['C@@H', 'C@H', 'N@H+', 'Nb', 'Ta', 'N', 'c', 'n', 'CH', 'O', 'C', 'P', 'S', 'Cl', 'nH', 's', 'Br', 'o', 'I', 'H', '*', 'F', 'Ca', 'Al', 'OH', 'Na', 'NH', 'Se', 'Co', 'Hg', 'As', 'Mg', 'Cu', 'Si', 'Au', 'Tc', 'B', 'Fe', 'Ge', 'Sm', 'Ru', 'V', 'Mo', 'He', 'Sb', 'Yb', 'Gd', 'Li', 'Cr', 'Ag', 'Fr', 'Ba', 'Pb', 'Y', 'Sr', 'Ga', 'Eu', 'Mn', 'Os', 'Tl', 'In', 'Sn', 'Ir', 'La', 'Lu', 'Cs', 'Ce', 'W', 'Zn', 'Be', 'Bi', 'U', 'Ni', 'Ho', 'Pt', 'Rb', 'K', 'SeH', 'TeH', 'Te', 'At', 'Re', 'Ra', 'Ti', 'SiH', 'se', 'pH', 'te', 'Ar', 'Xe', 'Kr', 'Cd', 'Pd', 'Rh', 'cH', 'p', 'Ne', 'Rn', 'LiH', 'Zr', 'AsH', 'Pr', 'Po', 'Tb'], key=lambda x: -len(x))
    def __init__(self):
        self.__atom_chars = ["\*", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl",
                      "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se",
                      "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb",
                      "Te", "I", "Xe", "Cs", "Ba", "La", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb",
                      "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh",
                      "Fl", "Mc", "Lv", "Ts", "Og", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm",
                      "Yb", "Lu", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "c",
                      "n", "s", "o", "se", "p", "te"]

        assert len(self.__atom_chars) == 118 + 8, len(self.__atom_chars)
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
        return [ i for smiles in _tokenize(input) for i in self._encode_token(smiles) ]

    def _encode_token(self, token):
        t, x = token
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
                    for i in self._encode_atoms(match.group("atoms")):
                        yield i
                if match.group("complex_atom"):
                    for i in self._encode_atoms(match.group("lhr")):
                        yield i
                    yield self.__modifier_chars.index(match.group("atom_modifier")) + offset
                    for i in self._encode_atoms(match.group("rhr")):
                        yield i
                offset += len(self.__atom_chars)
                if match.group("modifier"):
                    yield self.__modifier_chars.index(match.group("modifier")) + offset
                offset += len(self.__modifier_chars)
                if match.group("isotopes"):
                    yield offset
                offset += 1
                for i in range(0, 2):
                    if match.group("multiplicity_" + str(i)):
                        yield int(match.group("multiplicity_" + str(i))) + offset + self.__mults * i
            else:
                raise Exception("Could not encode atom", x)
        else:
            offset += len(self.__atom_chars) + len(self.__modifier_chars) + 1 + 2 * self.__mults
            if t == TokenType.BOND_TYPE or t == TokenType.EZSTEREO:
                yield self.__bond_chars.index(x)

            else:
                offset += len(self.__bond_chars)
                if t == TokenType.RING_NUM:
                    yield (x + offset - 1)
                else:
                    offset += 36
                    if t == TokenType.BRANCH_START or t == TokenType.BRANCH_END:
                        yield (self.__branch_chars.index(x))

    def _encode_atoms(self, group):
        for submatch in re.finditer(self.__atom_regex, group):
            atom = submatch.groups()[0]
            if atom == "*":
                atom = "\*"
            yield self.__atom_chars.index(atom)


class AtomOrdEncoder(Encoder):
    def __init__(self):
        __tokens = {
            1: ['*', '11B', '123I', '125I', '129Xe', '131I', '13C', '13C@', '13C@@', '13C@@H', '13C@H', '13CH', '13CH2',
                '13CH3', '13c', '13cH', '14C', '14CH3', '14c', '15N', '15NH', '15NH2', '15NH3', '15NH4', '15n', '15nH',
                '18F', '18O', '18OH', '197Au', '197Hg', '1CH', '1H', '201Tl', '203Hg', '208PbH2', '223Ra', '24O', '2H',
                '2NaH', '3H', '3He', '3NH4', '3NaH', '4He', '4NaH', '51V', '54Fe', '57Fe', '63Cu', '67Zn', '6Li', '6Na',
                '75Se', '99Tc', '9Be', 'Ac', 'Ag', 'Al', 'AlH', 'AlH2', 'AlH3', 'AlH4', 'Ar', 'As', 'AsH', 'At', 'Au',
                'AuH2', 'B', 'B@', 'B@@', 'BH', 'BH2', 'BH3', 'BH4', 'Ba', 'BaH', 'BaH2', 'Be', 'Bi', 'Bi0', 'BiH',
                'BiH2', 'BiH3', 'Br', 'Br0', 'Br1', 'Br2', 'Br3', 'Br4', 'Br5', 'Br6', 'Br7', 'Br8', 'Br9', 'BrH',
                'BrH2', 'C', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C@', 'C@0', 'C@2', 'C@3',
                'C@9', 'C@@', 'C@@1', 'C@@3', 'C@@5', 'C@@7', 'C@@H', 'C@@H0', 'C@@H1', 'C@@H2', 'C@@H4', 'C@@H6',
                'C@H', 'C@H0', 'C@H1', 'C@H2', 'C@H3', 'C@H4', 'CH', 'CH0', 'CH1', 'CH2', 'CH20', 'CH21', 'CH22',
                'CH23', 'CH24', 'CH25', 'CH26', 'CH27', 'CH28', 'CH29', 'CH3', 'CH30', 'CH31', 'CH32', 'CH33', 'CH34',
                'CH35', 'CH36', 'CH37', 'CH38', 'CH39', 'CH4', 'CH6', 'CH7', 'Ca', 'CaH', 'CaH2', 'Cd', 'Ce', 'Ce0',
                'Cl', 'Cl0', 'Cl1', 'Cl2', 'Cl3', 'Cl4', 'Cl5', 'Cl6', 'Cl7', 'Cl8', 'Cl9', 'ClH', 'ClH0', 'ClH2',
                'ClH3', 'ClH4', 'ClH5', 'ClH6', 'ClH7', 'ClH8', 'Co', 'CoH5', 'CoH6', 'Cr', 'Cs', 'CsH', 'Cu', 'CuH2',
                'CuH3', 'CuH4', 'Dy', 'Er', 'Eu', 'Eu1', 'F', 'F0', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8',
                'F9', 'FH', 'Fe', 'FeH2', 'FeH3', 'FeH4', 'FeH5', 'FeH6', 'Fr', 'Ga', 'GaH2', 'GaH3', 'Gd', 'Gd1', 'Ge',
                'GeH', 'GeH2', 'GeH3', 'GeH4', 'H', 'H0', 'H1', 'H2', 'H3', 'H4', 'H6', 'H7', 'H8', 'HH', 'He', 'HeH',
                'Hf', 'Hf2', 'Hg', 'Ho', 'I', 'I0', 'I1', 'I2', 'I7', 'IH', 'IH2', 'In', 'InH3', 'Ir', 'Ir0', 'IrH4',
                'K', 'KH', 'KH2', 'Kr', 'La', 'Li', 'LiH', 'Lu', 'Mg', 'MgH', 'MgH2', 'MgH4', 'Mn', 'MnH4', 'MnH5',
                'Mo', 'N', 'N0', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8', 'N9', 'N@', 'N@@', 'N@H', 'NH', 'NH0',
                'NH1', 'NH2', 'NH20', 'NH21', 'NH22', 'NH23', 'NH24', 'NH25', 'NH26', 'NH27', 'NH28', 'NH29', 'NH3',
                'NH4', 'NH5', 'NH6', 'NH7', 'Na', 'Na0', 'Na1', 'Na5', 'NaH', 'NaH2', 'Nb', 'Nd', 'Ne', 'Ni', 'NiH2',
                'O', 'O0', 'O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7', 'O8', 'O9', 'OH', 'OH0', 'OH1', 'OH2', 'OH3',
                'OH4', 'OH5', 'OH6', 'OH7', 'OH8', 'OH9', 'Os', 'P', 'P9', 'P@', 'P@@', 'PH', 'PH2', 'PH3', 'Pb', 'PbH',
                'PbH2', 'PbH4', 'Pd', 'Pm', 'Po', 'Pr', 'Pt', 'Pt0', 'PtH4', 'PtH60', 'Rb', 'RbH', 'Re', 'Re2', 'Rh',
                'RhH', 'Rn', 'Ru', 'RuH', 'RuH2', 'RuH6', 'S', 'S0', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8',
                'S@', 'S@@', 'SH', 'SH2', 'Sb', 'Sb1', 'SbH', 'SbH2', 'SbH3', 'Sc', 'Se', 'SeH', 'SeH2', 'Si', 'SiH',
                'SiH2', 'SiH3', 'SiH4', 'Sm', 'Sn', 'Sn0', 'SnH', 'SnH2', 'SnH3', 'SnH4', 'SnH60', 'Sr', 'SrH2', 'Ta',
                'Tb', 'Tc', 'Te', 'TeH', 'TeH2', 'Th', 'Ti', 'TiH4', 'Tl', 'TlH', 'Tm', 'U', 'V', 'V0', 'VH2', 'VH7',
                'VH80', 'W', 'W0', 'WH70', 'Xe', 'Y', 'Yb', 'Zn', 'ZnH2', 'ZnH4', 'Zr', 'c', 'c0', 'c1', 'c2', 'c3',
                'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'cH', 'cH0', 'cH1', 'cH2', 'cH3', 'cH4', 'cH5', 'cH6', 'cH7', 'cH8',
                'cH9', 'n', 'n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'nH', 'o', 'o0', 'o1', 'o2', 'o3', 'o6',
                'p', 'pH', 's', 's0', 's1', 's2', 's3', 's4', 's6', 's7', 's9', 'se', 'te'],
            5: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                29, 30, 31, 32, 33, 34, 35, 36], 4: [')'], 2: ['', '#', '.', '='], 6: ['/', '\\'], 3: ['(']}

        offsets = {t: int(sum(len(__tokens[t2]) for t2 in range(1, t)) * 1.05) for t in __tokens}

        self.lookup = {t: i+offsets[key] for key, values in __tokens.items() for (i, t) in enumerate(values)}

    @property
    def shape(self):
        raise (None, 1)

    def _encode_token(self, token):
        t, x = token
        if isinstance(x, str):
            x = re.sub(r":\d|\+\+?\d?|--?\d?", "", x.replace("[", "").replace("]", "")) # brush up token a bit
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



