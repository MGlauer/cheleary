from pysmiles.read_smiles import _tokenize, TokenType
import re
import numpy as np
from cheleary.registry import Registerable

_ENCODERS = dict()


class Encoder(Registerable):
    """
    Encodes data into a different format. If ID is specified, it should be unique amongst Encoders.
    """

    _REGISTRY = _ENCODERS

    @property
    def shape(self):
        raise NotImplementedError

    def run(self, input):
        raise NotImplementedError


class SmilesOneHotEncoder(Encoder):
    # atom_chars = sorted(['C@@H', 'C@H', 'N@H+', 'Nb', 'Ta', 'N', 'c', 'n', 'CH', 'O', 'C', 'P', 'S', 'Cl', 'nH', 's', 'Br', 'o', 'I', 'H', '*', 'F', 'Ca', 'Al', 'OH', 'Na', 'NH', 'Se', 'Co', 'Hg', 'As', 'Mg', 'Cu', 'Si', 'Au', 'Tc', 'B', 'Fe', 'Ge', 'Sm', 'Ru', 'V', 'Mo', 'He', 'Sb', 'Yb', 'Gd', 'Li', 'Cr', 'Ag', 'Fr', 'Ba', 'Pb', 'Y', 'Sr', 'Ga', 'Eu', 'Mn', 'Os', 'Tl', 'In', 'Sn', 'Ir', 'La', 'Lu', 'Cs', 'Ce', 'W', 'Zn', 'Be', 'Bi', 'U', 'Ni', 'Ho', 'Pt', 'Rb', 'K', 'SeH', 'TeH', 'Te', 'At', 'Re', 'Ra', 'Ti', 'SiH', 'se', 'pH', 'te', 'Ar', 'Xe', 'Kr', 'Cd', 'Pd', 'Rh', 'cH', 'p', 'Ne', 'Rn', 'LiH', 'Zr', 'AsH', 'Pr', 'Po', 'Tb'], key=lambda x: -len(x))
    def __init__(self):
        super(SmilesOneHotEncoder, self).__init__()
        self.__atom_chars = [
            "\*",
            "H",
            "He",
            "Li",
            "Be",
            "B",
            "C",
            "N",
            "O",
            "F",
            "Ne",
            "Na",
            "Mg",
            "Al",
            "Si",
            "P",
            "S",
            "Cl",
            "Ar",
            "K",
            "Ca",
            "Sc",
            "Ti",
            "V",
            "Cr",
            "Mn",
            "Fe",
            "Co",
            "Ni",
            "Cu",
            "Zn",
            "Ga",
            "Ge",
            "As",
            "Se",
            "Br",
            "Kr",
            "Rb",
            "Sr",
            "Y",
            "Zr",
            "Nb",
            "Mo",
            "Tc",
            "Ru",
            "Rh",
            "Pd",
            "Ag",
            "Cd",
            "In",
            "Sn",
            "Sb",
            "Te",
            "I",
            "Xe",
            "Cs",
            "Ba",
            "La",
            "Hf",
            "Ta",
            "W",
            "Re",
            "Os",
            "Ir",
            "Pt",
            "Au",
            "Hg",
            "Tl",
            "Pb",
            "Bi",
            "Po",
            "At",
            "Rn",
            "Fr",
            "Ra",
            "Ac",
            "Rf",
            "Db",
            "Sg",
            "Bh",
            "Hs",
            "Mt",
            "Ds",
            "Rg",
            "Cn",
            "Nh",
            "Fl",
            "Mc",
            "Lv",
            "Ts",
            "Og",
            "Ce",
            "Pr",
            "Nd",
            "Pm",
            "Sm",
            "Eu",
            "Gd",
            "Tb",
            "Dy",
            "Ho",
            "Er",
            "Tm",
            "Yb",
            "Lu",
            "Th",
            "Pa",
            "U",
            "Np",
            "Pu",
            "Am",
            "Cm",
            "Bk",
            "Cf",
            "Es",
            "Fm",
            "Md",
            "No",
            "Lr",
            "c",
            "n",
            "s",
            "o",
            "se",
            "p",
        ]

        assert len(self.__atom_chars) == 118 + 7, len(self.__atom_chars)
        self.__modifier_chars = ["+", "++", "-", "--", "@", "@@", "@+", "@@+", ":"]
        self.__bond_chars = [".", "-", "=", "#", "$", "/", "\\"]
        self.__branch_chars = ["(", ")"]
        self.__mults = 12
        self.__mult_regex = "|".join(map(str, range(1, self.__mults + 1)))
        self.__input_lenth = (
            len(self.__atom_chars)
            + len(self.__modifier_chars)
            + 1
            + 2 * self.__mults
            + len(self.__bond_chars)
            + 36
            + len(self.__branch_chars)
        )

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
            regex = re.compile(
                rf"^(?P<isotopes>\d+)?"
                rf"((?P<atoms>({self.__atom_regex})*)|(?P<complex_atom>({self.__complex_atom_regex})+))"
                rf"(?P<multiplicity_0>{self.__mult_regex})?"
                rf"(?P<modifier>{'|'.join(map(re.escape, self.__modifier_chars))})?"
                rf"(?P<multiplicity_1>{self.__mult_regex})?$"
            )
            for y in ["[", "]"]:
                x = x.replace(y, "")
            match = regex.match(x)
            if match is not None:
                offset = 0
                if match.group("atoms"):
                    self._encode_atoms(match.group("atoms"), indices)
                if match.group("complex_atom"):
                    self._encode_atoms(match.group("lhr"), indices)
                    indices.append(
                        self.__modifier_chars.index(match.group("atom_modifier"))
                        + offset
                    )
                    self._encode_atoms(match.group("rhr"), indices)
                offset += len(self.__atom_chars)
                if match.group("modifier"):
                    indices.append(
                        self.__modifier_chars.index(match.group("modifier")) + offset
                    )
                offset += len(self.__modifier_chars)
                if match.group("isotopes"):
                    indices.append(offset)
                offset += 1
                for i in range(0, 2):
                    if match.group("multiplicity_" + str(i)):
                        indices.append(
                            int(match.group("multiplicity_" + str(i)))
                            + offset
                            + self.__mults * i
                        )
            else:
                raise Exception("Could not encode atom", x)
        else:
            offset += (
                len(self.__atom_chars)
                + len(self.__modifier_chars)
                + 1
                + 2 * self.__mults
            )
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
    """Parses individual atoms and encodes them by their periodic number. 0 represents the `*` wild card. Bond types are
    enumerated and encoded accordingly."""

    _ID = "SAE"

    # atom_chars = sorted(['C@@H', 'C@H', 'N@H+', 'Nb', 'Ta', 'N', 'c', 'n', 'CH', 'O', 'C', 'P', 'S', 'Cl', 'nH', 's', 'Br', 'o', 'I', 'H', '*', 'F', 'Ca', 'Al', 'OH', 'Na', 'NH', 'Se', 'Co', 'Hg', 'As', 'Mg', 'Cu', 'Si', 'Au', 'Tc', 'B', 'Fe', 'Ge', 'Sm', 'Ru', 'V', 'Mo', 'He', 'Sb', 'Yb', 'Gd', 'Li', 'Cr', 'Ag', 'Fr', 'Ba', 'Pb', 'Y', 'Sr', 'Ga', 'Eu', 'Mn', 'Os', 'Tl', 'In', 'Sn', 'Ir', 'La', 'Lu', 'Cs', 'Ce', 'W', 'Zn', 'Be', 'Bi', 'U', 'Ni', 'Ho', 'Pt', 'Rb', 'K', 'SeH', 'TeH', 'Te', 'At', 'Re', 'Ra', 'Ti', 'SiH', 'se', 'pH', 'te', 'Ar', 'Xe', 'Kr', 'Cd', 'Pd', 'Rh', 'cH', 'p', 'Ne', 'Rn', 'LiH', 'Zr', 'AsH', 'Pr', 'Po', 'Tb'], key=lambda x: -len(x))
    def __init__(self):
        super(SmilesAtomEncoder, self).__init__()
        self.__atom_chars = [
            "\*",
            "H",
            "He",
            "Li",
            "Be",
            "B",
            "C",
            "N",
            "O",
            "F",
            "Ne",
            "Na",
            "Mg",
            "Al",
            "Si",
            "P",
            "S",
            "Cl",
            "Ar",
            "K",
            "Ca",
            "Sc",
            "Ti",
            "V",
            "Cr",
            "Mn",
            "Fe",
            "Co",
            "Ni",
            "Cu",
            "Zn",
            "Ga",
            "Ge",
            "As",
            "Se",
            "Br",
            "Kr",
            "Rb",
            "Sr",
            "Y",
            "Zr",
            "Nb",
            "Mo",
            "Tc",
            "Ru",
            "Rh",
            "Pd",
            "Ag",
            "Cd",
            "In",
            "Sn",
            "Sb",
            "Te",
            "I",
            "Xe",
            "Cs",
            "Ba",
            "La",
            "Hf",
            "Ta",
            "W",
            "Re",
            "Os",
            "Ir",
            "Pt",
            "Au",
            "Hg",
            "Tl",
            "Pb",
            "Bi",
            "Po",
            "At",
            "Rn",
            "Fr",
            "Ra",
            "Ac",
            "Rf",
            "Db",
            "Sg",
            "Bh",
            "Hs",
            "Mt",
            "Ds",
            "Rg",
            "Cn",
            "Nh",
            "Fl",
            "Mc",
            "Lv",
            "Ts",
            "Og",
            "Ce",
            "Pr",
            "Nd",
            "Pm",
            "Sm",
            "Eu",
            "Gd",
            "Tb",
            "Dy",
            "Ho",
            "Er",
            "Tm",
            "Yb",
            "Lu",
            "Th",
            "Pa",
            "U",
            "Np",
            "Pu",
            "Am",
            "Cm",
            "Bk",
            "Cf",
            "Es",
            "Fm",
            "Md",
            "No",
            "Lr",
            "c",
            "n",
            "s",
            "o",
            "se",
            "p",
            "te",
        ]

        assert len(self.__atom_chars) == 118 + 8, len(self.__atom_chars)
        self.__modifier_chars = ["+", "++", "-", "--", "@", "@@", "@+", "@@+", ":"]
        self.__bond_chars = [".", "-", "=", "#", "$", "/", "\\"]
        self.__branch_chars = ["(", ")", "[", "]"]
        self.__mults = 12
        self.__mult_regex = "|".join(map(str, range(1, self.__mults + 1)))
        self.__input_lenth = (
            len(self.__atom_chars)
            + len(self.__modifier_chars)
            + 1
            + 2 * self.__mults
            + len(self.__bond_chars)
            + 36
            + len(self.__branch_chars)
        )

        self.__atom_regex = rf"({'|'.join(self.__atom_chars)})"
        self.__complex_atom_regex = fr"(?P<lhr>{self.__atom_regex}+)(?P<atom_modifier>{'|'.join(map(re.escape, self.__modifier_chars))})(?P<rhr>{self.__atom_regex}+)"

    @property
    def shape(self):
        raise (None, self.__input_lenth)

    def run(self, input):
        return [i for smiles in _tokenize(input[1]) for i in self._encode_token(smiles)]

    d = {}

    def break_on_inconsistent(f):
        def inner(*args):
            self = args[0]
            x = tuple(args[1:])
            y = tuple(f(*args))
            if x in self.d:
                assert self.d[y] == x, f"d({y})={x} != {self.d[y]}"
            else:
                self.d[y] = x
            return y

        return inner

    @break_on_inconsistent
    def _encode_token(self, token):
        t, x = token
        offset = 0
        if t == TokenType.ATOM:
            regex = re.compile(
                rf"^(?P<isotopes>\d+)?"
                rf"((?P<atoms>({self.__atom_regex})*)|(?P<complex_atom>({self.__complex_atom_regex})+))"
                rf"(?P<multiplicity_0>{self.__mult_regex})?"
                rf"(?P<modifier>{'|'.join(map(re.escape, self.__modifier_chars))})?"
                rf"(?P<multiplicity_1>{self.__mult_regex})?$"
            )
            in_brackets = False
            if x[0] == "[":
                in_brackets = True
                for y in ["[", "]"]:
                    x = x.replace(y, "")
            yield offset
            match = regex.match(x)
            if match is not None:
                offset = 1
                if match.group("atoms"):
                    for i in self._encode_atoms(match.group("atoms")):
                        yield i + offset
                if match.group("complex_atom"):
                    for i in self._encode_atoms(match.group("lhr")):
                        yield i + offset
                    yield self.__modifier_chars.index(
                        match.group("atom_modifier")
                    ) + offset
                    for i in self._encode_atoms(match.group("rhr")):
                        yield i + offset
                offset += len(self.__atom_chars)
                if match.group("modifier"):
                    yield self.__modifier_chars.index(match.group("modifier")) + offset
                offset += len(self.__modifier_chars)
                if match.group("isotopes"):
                    yield offset
                offset += 1
                for i in range(0, 2):
                    if match.group("multiplicity_" + str(i)):
                        yield int(
                            match.group("multiplicity_" + str(i))
                        ) + offset + self.__mults * i
            else:
                raise Exception("Could not encode atom", x)
            offset += (
                len(self.__atom_chars)
                + len(self.__modifier_chars)
                + 1
                + 2 * self.__mults
            )
            if in_brackets:
                yield offset
            offset += 1
        else:
            offset += (
                len(self.__atom_chars)
                + len(self.__modifier_chars)
                + 1
                + 2 * self.__mults
                + 2
            )
            if t == TokenType.BOND_TYPE or t == TokenType.EZSTEREO:
                yield self.__bond_chars.index(x) + offset

            else:
                offset += len(self.__bond_chars)
                if t == TokenType.RING_NUM:
                    yield x + offset - 1
                else:
                    offset += 36
                    if t == TokenType.BRANCH_START or t == TokenType.BRANCH_END:
                        yield self.__branch_chars.index(x) + offset

    def _encode_atoms(self, group):
        results = False
        for submatch in re.finditer(self.__atom_regex, group):
            results = True
            atom = submatch.groups()[0]
            if atom == "*":
                atom = "\*"
            yield self.__atom_chars.index(atom)
        if not results:
            raise Exception("Could not encode", group)


class AtomOrdEncoder(Encoder):
    ID = "AOE"

    def __init__(self):
        super(AtomOrdEncoder, self).__init__()
        __tokens = {
            1: [
                "*",
                "11B",
                "123I",
                "125I",
                "129Xe",
                "131I",
                "13C",
                "13C@",
                "13C@@",
                "13C@@H",
                "13C@H",
                "13CH",
                "13CH2",
                "13CH3",
                "13c",
                "13cH",
                "14C",
                "14CH3",
                "14c",
                "15N",
                "15NH",
                "15NH2",
                "15NH3",
                "15NH4",
                "15n",
                "15nH",
                "18F",
                "18O",
                "18OH",
                "197Au",
                "197Hg",
                "1CH",
                "1H",
                "201Tl",
                "203Hg",
                "208PbH2",
                "223Ra",
                "24O",
                "2H",
                "2NaH",
                "3H",
                "3He",
                "3NH4",
                "3NaH",
                "4He",
                "4NaH",
                "51V",
                "54Fe",
                "57Fe",
                "63Cu",
                "67Zn",
                "6Li",
                "6Na",
                "75Se",
                "99Tc",
                "9Be",
                "Ac",
                "Ag",
                "Al",
                "AlH",
                "AlH2",
                "AlH3",
                "AlH4",
                "Ar",
                "As",
                "AsH",
                "At",
                "Au",
                "AuH2",
                "B",
                "B@",
                "B@@",
                "BH",
                "BH2",
                "BH3",
                "BH4",
                "Ba",
                "BaH",
                "BaH2",
                "Be",
                "Bi",
                "Bi0",
                "BiH",
                "BiH2",
                "BiH3",
                "Br",
                "Br0",
                "Br1",
                "Br2",
                "Br3",
                "Br4",
                "Br5",
                "Br6",
                "Br7",
                "Br8",
                "Br9",
                "BrH",
                "BrH2",
                "C",
                "C0",
                "C1",
                "C2",
                "C3",
                "C4",
                "C5",
                "C6",
                "C7",
                "C8",
                "C9",
                "C@",
                "C@0",
                "C@2",
                "C@3",
                "C@9",
                "C@@",
                "C@@1",
                "C@@3",
                "C@@5",
                "C@@7",
                "C@@H",
                "C@@H0",
                "C@@H1",
                "C@@H2",
                "C@@H4",
                "C@@H6",
                "C@H",
                "C@H0",
                "C@H1",
                "C@H2",
                "C@H3",
                "C@H4",
                "CH",
                "CH0",
                "CH1",
                "CH2",
                "CH20",
                "CH21",
                "CH22",
                "CH23",
                "CH24",
                "CH25",
                "CH26",
                "CH27",
                "CH28",
                "CH29",
                "CH3",
                "CH30",
                "CH31",
                "CH32",
                "CH33",
                "CH34",
                "CH35",
                "CH36",
                "CH37",
                "CH38",
                "CH39",
                "CH4",
                "CH6",
                "CH7",
                "Ca",
                "CaH",
                "CaH2",
                "Cd",
                "Ce",
                "Ce0",
                "Cl",
                "Cl0",
                "Cl1",
                "Cl2",
                "Cl3",
                "Cl4",
                "Cl5",
                "Cl6",
                "Cl7",
                "Cl8",
                "Cl9",
                "ClH",
                "ClH0",
                "ClH2",
                "ClH3",
                "ClH4",
                "ClH5",
                "ClH6",
                "ClH7",
                "ClH8",
                "Co",
                "CoH5",
                "CoH6",
                "Cr",
                "Cs",
                "CsH",
                "Cu",
                "CuH2",
                "CuH3",
                "CuH4",
                "Dy",
                "Er",
                "Eu",
                "Eu1",
                "F",
                "F0",
                "F1",
                "F2",
                "F3",
                "F4",
                "F5",
                "F6",
                "F7",
                "F8",
                "F9",
                "FH",
                "Fe",
                "FeH2",
                "FeH3",
                "FeH4",
                "FeH5",
                "FeH6",
                "Fr",
                "Ga",
                "GaH2",
                "GaH3",
                "Gd",
                "Gd1",
                "Ge",
                "GeH",
                "GeH2",
                "GeH3",
                "GeH4",
                "H",
                "H0",
                "H1",
                "H2",
                "H3",
                "H4",
                "H6",
                "H7",
                "H8",
                "HH",
                "He",
                "HeH",
                "Hf",
                "Hf2",
                "Hg",
                "Ho",
                "I",
                "I0",
                "I1",
                "I2",
                "I7",
                "IH",
                "IH2",
                "In",
                "InH3",
                "Ir",
                "Ir0",
                "IrH4",
                "K",
                "KH",
                "KH2",
                "Kr",
                "La",
                "Li",
                "LiH",
                "Lu",
                "Mg",
                "MgH",
                "MgH2",
                "MgH4",
                "Mn",
                "MnH4",
                "MnH5",
                "Mo",
                "N",
                "N0",
                "N1",
                "N2",
                "N3",
                "N4",
                "N5",
                "N6",
                "N7",
                "N8",
                "N9",
                "N@",
                "N@@",
                "N@H",
                "NH",
                "NH0",
                "NH1",
                "NH2",
                "NH20",
                "NH21",
                "NH22",
                "NH23",
                "NH24",
                "NH25",
                "NH26",
                "NH27",
                "NH28",
                "NH29",
                "NH3",
                "NH4",
                "NH5",
                "NH6",
                "NH7",
                "Na",
                "Na0",
                "Na1",
                "Na5",
                "NaH",
                "NaH2",
                "Nb",
                "Nd",
                "Ne",
                "Ni",
                "NiH2",
                "O",
                "O0",
                "O1",
                "O2",
                "O3",
                "O4",
                "O5",
                "O6",
                "O7",
                "O8",
                "O9",
                "OH",
                "OH0",
                "OH1",
                "OH2",
                "OH3",
                "OH4",
                "OH5",
                "OH6",
                "OH7",
                "OH8",
                "OH9",
                "Os",
                "P",
                "P9",
                "P@",
                "P@@",
                "PH",
                "PH2",
                "PH3",
                "Pb",
                "PbH",
                "PbH2",
                "PbH4",
                "Pd",
                "Pm",
                "Po",
                "Pr",
                "Pt",
                "Pt0",
                "PtH4",
                "PtH60",
                "Rb",
                "RbH",
                "Re",
                "Re2",
                "Rh",
                "RhH",
                "Rn",
                "Ru",
                "RuH",
                "RuH2",
                "RuH6",
                "S",
                "S0",
                "S1",
                "S2",
                "S3",
                "S4",
                "S5",
                "S6",
                "S7",
                "S8",
                "S@",
                "S@@",
                "SH",
                "SH2",
                "Sb",
                "Sb1",
                "SbH",
                "SbH2",
                "SbH3",
                "Sc",
                "Se",
                "SeH",
                "SeH2",
                "Si",
                "SiH",
                "SiH2",
                "SiH3",
                "SiH4",
                "Sm",
                "Sn",
                "Sn0",
                "SnH",
                "SnH2",
                "SnH3",
                "SnH4",
                "SnH60",
                "Sr",
                "SrH2",
                "Ta",
                "Tb",
                "Tc",
                "Te",
                "TeH",
                "TeH2",
                "Th",
                "Ti",
                "TiH4",
                "Tl",
                "TlH",
                "Tm",
                "U",
                "V",
                "V0",
                "VH2",
                "VH7",
                "VH80",
                "W",
                "W0",
                "WH70",
                "Xe",
                "Y",
                "Yb",
                "Zn",
                "ZnH2",
                "ZnH4",
                "Zr",
                "c",
                "c0",
                "c1",
                "c2",
                "c3",
                "c4",
                "c5",
                "c6",
                "c7",
                "c8",
                "c9",
                "cH",
                "cH0",
                "cH1",
                "cH2",
                "cH3",
                "cH4",
                "cH5",
                "cH6",
                "cH7",
                "cH8",
                "cH9",
                "n",
                "n0",
                "n1",
                "n2",
                "n3",
                "n4",
                "n5",
                "n6",
                "n7",
                "nH",
                "o",
                "o0",
                "o1",
                "o2",
                "o3",
                "o6",
                "p",
                "pH",
                "s",
                "s0",
                "s1",
                "s2",
                "s3",
                "s4",
                "s6",
                "s7",
                "s9",
                "se",
                "te",
            ],
            5: [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
                28,
                29,
                30,
                31,
                32,
                33,
                34,
                35,
                36,
            ],
            4: [")"],
            2: ["", "#", ".", "="],
            6: ["/", "\\"],
            3: ["("],
        }

        offsets = {
            t: int(sum(len(__tokens[t2]) for t2 in range(1, t)) * 1.05)
            for t in __tokens
        }

        self.lookup = {
            t: i + offsets[key]
            for key, values in __tokens.items()
            for (i, t) in enumerate(values)
        }

    @property
    def shape(self):
        raise (None, 1)

    def _encode_token(self, token):
        t, x = token
        if isinstance(x, str):
            x = re.sub(
                r":\d|\+\+?\d?|--?\d?", "", x.replace("[", "").replace("]", "")
            )  # brush up token a bit
        return self.lookup[x]

    def run(self, input):
        return [self._encode_token(smiles) for smiles in _tokenize(input)]


class CharacterOrdEncoder(Encoder):
    _ID = "COE"

    @property
    def shape(self):
        raise (None, 1)

    def run(self, input):
        return np.asarray([ord(c) for c in input[1]])


class ChebiClassEncoder(Encoder):
    """Encodes chebi ids according to a fixed schema (see `ChebiClassEncoder.__CLASSES`)."""

    _ID = "CCE"
    __CLASSES = {
        "CHEBI:37533": 0,
        "CHEBI:25741": 1,
        "CHEBI:83822": 2,
        "CHEBI:35343": 3,
        "CHEBI:76170": 4,
        "CHEBI:26519": 5,
        "CHEBI:26819": 6,
        "CHEBI:17517": 7,
        "CHEBI:83565": 8,
        "CHEBI:58958": 9,
        "CHEBI:38106": 10,
        "CHEBI:134179": 11,
        "CHEBI:35294": 12,
        "CHEBI:35972": 13,
        "CHEBI:23824": 14,
        "CHEBI:50511": 15,
        "CHEBI:77636": 16,
        "CHEBI:15693": 17,
        "CHEBI:35189": 18,
        "CHEBI:26878": 19,
        "CHEBI:83273": 20,
        "CHEBI:58342": 21,
        "CHEBI:51683": 22,
        "CHEBI:61777": 23,
        "CHEBI:51151": 24,
        "CHEBI:27150": 25,
        "CHEBI:46955": 26,
        "CHEBI:38179": 27,
        "CHEBI:18035": 28,
        "CHEBI:15841": 29,
        "CHEBI:59777": 30,
        "CHEBI:35871": 31,
        "CHEBI:35267": 32,
        "CHEBI:27024": 33,
        "CHEBI:36141": 34,
        "CHEBI:18379": 35,
        "CHEBI:25810": 36,
        "CHEBI:50018": 37,
        "CHEBI:16460": 38,
        "CHEBI:33482": 39,
        "CHEBI:37739": 40,
        "CHEBI:74927": 41,
        "CHEBI:33572": 42,
        "CHEBI:59737": 43,
        "CHEBI:26421": 44,
        "CHEBI:60834": 45,
        "CHEBI:33692": 46,
        "CHEBI:33240": 47,
        "CHEBI:59835": 48,
        "CHEBI:22693": 49,
        "CHEBI:22712": 50,
        "CHEBI:36054": 51,
        "CHEBI:39206": 52,
        "CHEBI:62941": 53,
        "CHEBI:25413": 54,
        "CHEBI:23666": 55,
        "CHEBI:57643": 56,
        "CHEBI:33566": 57,
        "CHEBI:35281": 58,
        "CHEBI:83264": 59,
        "CHEBI:37407": 60,
        "CHEBI:62732": 61,
        "CHEBI:24689": 62,
        "CHEBI:2468": 63,
        "CHEBI:35716": 64,
        "CHEBI:38777": 65,
        "CHEBI:37734": 66,
        "CHEBI:23451": 67,
        "CHEBI:59266": 68,
        "CHEBI:35571": 69,
        "CHEBI:17855": 70,
        "CHEBI:72544": 71,
        "CHEBI:63436": 72,
        "CHEBI:35366": 73,
        "CHEBI:46874": 74,
        "CHEBI:73474": 75,
        "CHEBI:25248": 76,
        "CHEBI:22888": 77,
        "CHEBI:47018": 78,
        "CHEBI:38686": 79,
        "CHEBI:38164": 80,
        "CHEBI:38716": 81,
        "CHEBI:37948": 82,
        "CHEBI:37578": 83,
        "CHEBI:35692": 84,
        "CHEBI:46774": 85,
        "CHEBI:62937": 86,
        "CHEBI:38835": 87,
        "CHEBI:18154": 88,
        "CHEBI:35313": 89,
        "CHEBI:23906": 90,
        "CHEBI:25697": 91,
        "CHEBI:75885": 92,
        "CHEBI:47778": 93,
        "CHEBI:33859": 94,
        "CHEBI:25513": 95,
        "CHEBI:52575": 96,
        "CHEBI:25000": 97,
        "CHEBI:26347": 98,
        "CHEBI:79346": 99,
        "CHEBI:51702": 100,
        "CHEBI:23449": 101,
        "CHEBI:28965": 102,
        "CHEBI:46850": 103,
        "CHEBI:23665": 104,
        "CHEBI:46761": 105,
        "CHEBI:24835": 106,
        "CHEBI:24385": 107,
        "CHEBI:35992": 108,
        "CHEBI:25754": 109,
        "CHEBI:25477": 110,
        "CHEBI:28963": 111,
        "CHEBI:33447": 112,
        "CHEBI:35693": 113,
        "CHEBI:61778": 114,
        "CHEBI:59412": 115,
        "CHEBI:38976": 116,
        "CHEBI:15734": 117,
        "CHEBI:32952": 118,
        "CHEBI:33892": 119,
        "CHEBI:48975": 120,
        "CHEBI:60926": 121,
        "CHEBI:46812": 122,
        "CHEBI:78799": 123,
        "CHEBI:37143": 124,
        "CHEBI:35505": 125,
        "CHEBI:35259": 126,
        "CHEBI:24922": 127,
        "CHEBI:38131": 128,
        "CHEBI:23849": 129,
        "CHEBI:22702": 130,
        "CHEBI:35786": 131,
        "CHEBI:33273": 132,
        "CHEBI:64365": 133,
        "CHEBI:61910": 134,
        "CHEBI:59202": 135,
        "CHEBI:26020": 136,
        "CHEBI:61655": 137,
        "CHEBI:35903": 138,
        "CHEBI:36976": 139,
        "CHEBI:35356": 140,
        "CHEBI:35284": 141,
        "CHEBI:36807": 142,
        "CHEBI:38757": 143,
        "CHEBI:35902": 144,
        "CHEBI:28874": 145,
        "CHEBI:65321": 146,
        "CHEBI:33242": 147,
        "CHEBI:46848": 148,
        "CHEBI:32877": 149,
        "CHEBI:37141": 150,
        "CHEBI:140325": 151,
        "CHEBI:22562": 152,
        "CHEBI:22645": 153,
        "CHEBI:33709": 154,
        "CHEBI:63367": 155,
        "CHEBI:24921": 156,
        "CHEBI:36309": 157,
        "CHEBI:51276": 158,
        "CHEBI:23697": 159,
        "CHEBI:26455": 160,
        "CHEBI:24698": 161,
        "CHEBI:26799": 162,
        "CHEBI:17761": 163,
        "CHEBI:78840": 164,
        "CHEBI:46845": 165,
        "CHEBI:36314": 166,
        "CHEBI:48591": 167,
        "CHEBI:25529": 168,
        "CHEBI:131927": 169,
        "CHEBI:25703": 170,
        "CHEBI:79020": 171,
        "CHEBI:22723": 172,
        "CHEBI:36059": 173,
        "CHEBI:17984": 174,
        "CHEBI:37485": 175,
        "CHEBI:77632": 176,
        "CHEBI:22718": 177,
        "CHEBI:16038": 178,
        "CHEBI:38032": 179,
        "CHEBI:68489": 180,
        "CHEBI:36688": 181,
        "CHEBI:26707": 182,
        "CHEBI:37240": 183,
        "CHEBI:18303": 184,
        "CHEBI:38298": 185,
        "CHEBI:24129": 186,
        "CHEBI:23132": 187,
        "CHEBI:22315": 188,
        "CHEBI:25106": 189,
        "CHEBI:49172": 190,
        "CHEBI:63161": 191,
        "CHEBI:51149": 192,
        "CHEBI:33721": 193,
        "CHEBI:53339": 194,
        "CHEBI:26714": 195,
        "CHEBI:23217": 196,
        "CHEBI:83575": 197,
        "CHEBI:22475": 198,
        "CHEBI:15705": 199,
        "CHEBI:26605": 200,
        "CHEBI:63563": 201,
        "CHEBI:36700": 202,
        "CHEBI:38785": 203,
        "CHEBI:58945": 204,
        "CHEBI:35358": 205,
        "CHEBI:35186": 206,
        "CHEBI:63551": 207,
        "CHEBI:36785": 208,
        "CHEBI:50996": 209,
        "CHEBI:26712": 210,
        "CHEBI:36885": 211,
        "CHEBI:33637": 212,
        "CHEBI:27325": 213,
        "CHEBI:47788": 214,
        "CHEBI:139051": 215,
        "CHEBI:22331": 216,
        "CHEBI:33641": 217,
        "CHEBI:35213": 218,
        "CHEBI:22484": 219,
        "CHEBI:38093": 220,
        "CHEBI:61902": 221,
        "CHEBI:47622": 222,
        "CHEBI:26401": 223,
        "CHEBI:25384": 224,
        "CHEBI:37581": 225,
        "CHEBI:68452": 226,
        "CHEBI:47857": 227,
        "CHEBI:22160": 228,
        "CHEBI:22727": 229,
        "CHEBI:64708": 230,
        "CHEBI:38180": 231,
        "CHEBI:46770": 232,
        "CHEBI:35618": 233,
        "CHEBI:27116": 234,
        "CHEBI:48470": 235,
        "CHEBI:25036": 236,
        "CHEBI:38261": 237,
        "CHEBI:72010": 238,
        "CHEBI:50523": 239,
        "CHEBI:26399": 240,
        "CHEBI:35924": 241,
        "CHEBI:36916": 242,
        "CHEBI:64459": 243,
        "CHEBI:134396": 244,
        "CHEBI:38313": 245,
        "CHEBI:37909": 246,
        "CHEBI:26658": 247,
        "CHEBI:24531": 248,
        "CHEBI:24302": 249,
        "CHEBI:52782": 250,
        "CHEBI:24315": 251,
        "CHEBI:35276": 252,
        "CHEBI:35868": 253,
        "CHEBI:23403": 254,
        "CHEBI:35757": 255,
        "CHEBI:83812": 256,
        "CHEBI:59238": 257,
        "CHEBI:26979": 258,
        "CHEBI:16670": 259,
        "CHEBI:51959": 260,
        "CHEBI:21731": 261,
        "CHEBI:35753": 262,
        "CHEBI:46952": 263,
        "CHEBI:25961": 264,
        "CHEBI:51277": 265,
        "CHEBI:28892": 266,
        "CHEBI:74222": 267,
        "CHEBI:33676": 268,
        "CHEBI:25830": 269,
        "CHEBI:38193": 270,
        "CHEBI:24402": 271,
        "CHEBI:51718": 272,
        "CHEBI:57560": 273,
        "CHEBI:35274": 274,
        "CHEBI:35381": 275,
        "CHEBI:35406": 276,
        "CHEBI:37668": 277,
        "CHEBI:61697": 278,
        "CHEBI:24400": 279,
        "CHEBI:23677": 280,
        "CHEBI:33741": 281,
        "CHEBI:46940": 282,
        "CHEBI:51751": 283,
        "CHEBI:33658": 284,
        "CHEBI:23003": 285,
        "CHEBI:36130": 286,
        "CHEBI:24828": 287,
        "CHEBI:36786": 288,
        "CHEBI:33839": 289,
        "CHEBI:26144": 290,
        "CHEBI:33860": 291,
        "CHEBI:38260": 292,
        "CHEBI:64612": 293,
        "CHEBI:22480": 294,
        "CHEBI:38295": 295,
        "CHEBI:3992": 296,
        "CHEBI:36313": 297,
        "CHEBI:29067": 298,
        "CHEBI:50918": 299,
        "CHEBI:35681": 300,
        "CHEBI:131871": 301,
        "CHEBI:136889": 302,
        "CHEBI:23117": 303,
        "CHEBI:50492": 304,
        "CHEBI:48030": 305,
        "CHEBI:62643": 306,
        "CHEBI:36526": 307,
        "CHEBI:35683": 308,
        "CHEBI:38338": 309,
        "CHEBI:26822": 310,
        "CHEBI:17387": 311,
        "CHEBI:38445": 312,
        "CHEBI:36820": 313,
        "CHEBI:33853": 314,
        "CHEBI:27369": 315,
        "CHEBI:22750": 316,
        "CHEBI:33576": 317,
        "CHEBI:83403": 318,
        "CHEBI:33424": 319,
        "CHEBI:33299": 320,
        "CHEBI:35727": 321,
        "CHEBI:26407": 322,
        "CHEBI:24373": 323,
        "CHEBI:36699": 324,
        "CHEBI:46640": 325,
        "CHEBI:25716": 326,
        "CHEBI:64583": 327,
        "CHEBI:24654": 328,
        "CHEBI:26191": 329,
        "CHEBI:33552": 330,
        "CHEBI:47923": 331,
        "CHEBI:37175": 332,
        "CHEBI:50994": 333,
        "CHEBI:38530": 334,
        "CHEBI:26208": 335,
        "CHEBI:17478": 336,
        "CHEBI:25676": 337,
        "CHEBI:25872": 338,
        "CHEBI:48901": 339,
        "CHEBI:33296": 340,
        "CHEBI:24026": 341,
        "CHEBI:23066": 342,
        "CHEBI:139592": 343,
        "CHEBI:33653": 344,
        "CHEBI:26776": 345,
        "CHEBI:23232": 346,
        "CHEBI:37947": 347,
        "CHEBI:24780": 348,
        "CHEBI:35741": 349,
        "CHEBI:36233": 350,
        "CHEBI:36683": 351,
        "CHEBI:17087": 352,
        "CHEBI:46942": 353,
        "CHEBI:36684": 354,
        "CHEBI:64482": 355,
        "CHEBI:26816": 356,
        "CHEBI:26766": 357,
        "CHEBI:33838": 358,
        "CHEBI:22928": 359,
        "CHEBI:46867": 360,
        "CHEBI:36132": 361,
        "CHEBI:26912": 362,
        "CHEBI:61379": 363,
        "CHEBI:26562": 364,
        "CHEBI:26739": 365,
        "CHEBI:13248": 366,
        "CHEBI:38163": 367,
        "CHEBI:23114": 368,
        "CHEBI:25409": 369,
        "CHEBI:20857": 370,
        "CHEBI:51006": 371,
        "CHEBI:62733": 372,
        "CHEBI:50699": 373,
        "CHEBI:50753": 374,
        "CHEBI:23990": 375,
        "CHEBI:24860": 376,
        "CHEBI:26151": 377,
        "CHEBI:24471": 378,
        "CHEBI:50995": 379,
        "CHEBI:33694": 380,
        "CHEBI:33720": 381,
        "CHEBI:76224": 382,
        "CHEBI:22798": 383,
        "CHEBI:51069": 384,
        "CHEBI:38830": 385,
        "CHEBI:37022": 386,
        "CHEBI:51614": 387,
        "CHEBI:51569": 388,
        "CHEBI:35348": 389,
        "CHEBI:30879": 390,
        "CHEBI:37667": 391,
        "CHEBI:35819": 392,
        "CHEBI:26469": 393,
        "CHEBI:16247": 394,
        "CHEBI:61109": 395,
        "CHEBI:35341": 396,
        "CHEBI:51689": 397,
        "CHEBI:33551": 398,
        "CHEBI:48888": 399,
        "CHEBI:23213": 400,
        "CHEBI:24868": 401,
        "CHEBI:83821": 402,
        "CHEBI:26410": 403,
        "CHEBI:25704": 404,
        "CHEBI:23981": 405,
        "CHEBI:22715": 406,
        "CHEBI:51867": 407,
        "CHEBI:63353": 408,
        "CHEBI:24834": 409,
        "CHEBI:47880": 410,
        "CHEBI:63423": 411,
        "CHEBI:50525": 412,
        "CHEBI:63944": 413,
        "CHEBI:27136": 414,
        "CHEBI:65323": 415,
        "CHEBI:76578": 416,
        "CHEBI:33570": 417,
        "CHEBI:58946": 418,
        "CHEBI:83811": 419,
        "CHEBI:26513": 420,
        "CHEBI:22728": 421,
        "CHEBI:35990": 422,
        "CHEBI:61355": 423,
        "CHEBI:33259": 424,
        "CHEBI:38700": 425,
        "CHEBI:38771": 426,
        "CHEBI:33823": 427,
        "CHEBI:26195": 428,
        "CHEBI:36358": 429,
        "CHEBI:33558": 430,
        "CHEBI:38958": 431,
        "CHEBI:22483": 432,
        "CHEBI:33563": 433,
        "CHEBI:26893": 434,
        "CHEBI:18946": 435,
        "CHEBI:24062": 436,
        "CHEBI:32955": 437,
        "CHEBI:39447": 438,
        "CHEBI:35241": 439,
        "CHEBI:36709": 440,
        "CHEBI:26873": 441,
        "CHEBI:23424": 442,
        "CHEBI:35748": 443,
        "CHEBI:24397": 444,
        "CHEBI:35350": 445,
        "CHEBI:33702": 446,
        "CHEBI:35789": 447,
        "CHEBI:16646": 448,
        "CHEBI:36615": 449,
        "CHEBI:35507": 450,
        "CHEBI:38443": 451,
        "CHEBI:24436": 452,
        "CHEBI:83925": 453,
        "CHEBI:36315": 454,
        "CHEBI:22916": 455,
        "CHEBI:36834": 456,
        "CHEBI:72823": 457,
        "CHEBI:22485": 458,
        "CHEBI:28868": 459,
        "CHEBI:35238": 460,
        "CHEBI:26004": 461,
        "CHEBI:29347": 462,
        "CHEBI:35295": 463,
        "CHEBI:38418": 464,
        "CHEBI:35715": 465,
        "CHEBI:25985": 466,
        "CHEBI:25235": 467,
        "CHEBI:24697": 468,
        "CHEBI:36087": 469,
        "CHEBI:38314": 470,
        "CHEBI:29348": 471,
        "CHEBI:33554": 472,
        "CHEBI:59268": 473,
        "CHEBI:33184": 474,
        "CHEBI:47779": 475,
        "CHEBI:61120": 476,
        "CHEBI:36094": 477,
        "CHEBI:25508": 478,
        "CHEBI:36244": 479,
        "CHEBI:25698": 480,
        "CHEBI:36520": 481,
        "CHEBI:16385": 482,
        "CHEBI:24866": 483,
        "CHEBI:37010": 484,
        "CHEBI:76567": 485,
        "CHEBI:58944": 486,
        "CHEBI:26961": 487,
        "CHEBI:24897": 488,
        "CHEBI:51454": 489,
        "CHEBI:24913": 490,
        "CHEBI:36836": 491,
        "CHEBI:33642": 492,
        "CHEBI:24676": 493,
        "CHEBI:33704": 494,
        "CHEBI:47916": 495,
        "CHEBI:26188": 496,
        "CHEBI:27093": 497,
        "CHEBI:59554": 498,
        "CHEBI:51681": 499,
    }

    def run(self, input):
        return [
            1 if i == self.__CLASSES[input] else 0 for i in range(len(self.__CLASSES))
        ]


class PosEncoder(Encoder):
    _ID = "POS"

    def run(self, input):
        return [i for i in range(len(input[2:])) if input[2 + i]]


class IntEncoder(Encoder):
    """Casts a list into a list of integers (`[int(c) for c in input]`)"""

    _ID = "IE"

    @property
    def shape(self):
        raise (None, 1)

    def run(self, input):
        return [int(c) for c in input[2:]]


class FingerprintEncoder(Encoder):
    _ID = "FE"

    def run(self, input):
        try:
            import rdkit
            from rdkit import Chem
        except Exception as e:
            raise e
        else:
            mol = Chem.MolFromSmiles(input[1])
            if mol:
                return list(map(bool, Chem.RDKFingerprint(mol, fpSize=1024)))


class ReadyFingerprintEncoder(Encoder):
    _ID = "RFE"

    def run(self, input):
        return np.asarray(tuple(map(int, input[3:])))


class ChebiLogitEncode(Encoder):
    _ID = "CLE"

    d = {}
    max_index = -1

    def run(self, input):
        cls = input[0]
        n = np.zeros(100)
        try:
            idx = self.d[cls]
        except KeyError:
            self.max_index += 1
            self.d[cls] = idx = self.max_index
        n[idx] = 1
        return n
