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



