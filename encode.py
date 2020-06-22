from pysmiles.read_smiles import _tokenize, TokenType
import re
import numpy as np
import tensorflow as tf

#atom_chars = sorted(['C@@H', 'C@H', 'N@H+', 'Nb', 'Ta', 'N', 'c', 'n', 'CH', 'O', 'C', 'P', 'S', 'Cl', 'nH', 's', 'Br', 'o', 'I', 'H', '*', 'F', 'Ca', 'Al', 'OH', 'Na', 'NH', 'Se', 'Co', 'Hg', 'As', 'Mg', 'Cu', 'Si', 'Au', 'Tc', 'B', 'Fe', 'Ge', 'Sm', 'Ru', 'V', 'Mo', 'He', 'Sb', 'Yb', 'Gd', 'Li', 'Cr', 'Ag', 'Fr', 'Ba', 'Pb', 'Y', 'Sr', 'Ga', 'Eu', 'Mn', 'Os', 'Tl', 'In', 'Sn', 'Ir', 'La', 'Lu', 'Cs', 'Ce', 'W', 'Zn', 'Be', 'Bi', 'U', 'Ni', 'Ho', 'Pt', 'Rb', 'K', 'SeH', 'TeH', 'Te', 'At', 'Re', 'Ra', 'Ti', 'SiH', 'se', 'pH', 'te', 'Ar', 'Xe', 'Kr', 'Cd', 'Pd', 'Rh', 'cH', 'p', 'Ne', 'Rn', 'LiH', 'Zr', 'AsH', 'Pr', 'Po', 'Tb'], key=lambda x: -len(x))
atom_chars = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"]

assert len(atom_chars)==103
modifier_chars = ['+', '++', '-', '--', '@', '@@', '@+', '@@+']
bond_chars = [".", "-", "=", "#", "$", ":", "/", "\\"]
branch_chars = ["(", ")"]
mults = 9
isotopes = ['1', '123', '125', '129', '13', '131', '14', '15', '18', '197', '2', '201', '203', '223', '3', '4', '51', '63', '67', '75', '9', '99']
input_lenth = len(atom_chars)  + len(modifier_chars) + len(isotopes) + 2*mults + len(bond_chars) + 36 + len(branch_chars)

atom_regex = r"(([A-Z][a-z]*)|[cnso]|se)"
complex_atom_regex = fr"(?P<lhr>{atom_regex}+)(?P<atom_modifier>{'|'.join(map(re.escape, modifier_chars))})(?P<rhr>{atom_regex}+)"

def encode_smiles(string):
    return [encode(smiles) for smiles in _tokenize(string)]

def encode_atoms(group, indices):
    for submatch in re.finditer(atom_regex, group):
        atom = submatch.groups(0)
        if atom not in atom_chars:
            atom_chars.append(atom)
        indices.append(atom_chars.index(atom))

def encode(token):
    v = np.zeros(input_lenth, dtype=np.bool)
    t, x = token
    indices = []
    offset = 0
    if t == TokenType.ATOM:
        regex = re.compile(rf"^(?P<isotopes>{'|'.join(map(re.escape, isotopes))})?((?P<atoms>({atom_regex})*)|(?P<complex_atom>({complex_atom_regex})+))(?P<multiplicity_0>[1-{mults}])?(?P<modifier>{'|'.join(map(re.escape, modifier_chars))})?(?P<multiplicity_1>[1-{mults}])?$")
        for y in ["[", "]"]:
            x = x.replace(y, "")
        match = regex.match(x)
        if match is not None:
            offset = 0
            if match.group("atoms"):
                encode_atoms(match.group("atoms"), indices)
            if match.group("complex_atom"):
                encode_atoms(match.group("lhr"), indices)
                indices.append(modifier_chars.index(match.group("atom_modifier")) + offset)
                encode_atoms(match.group("rhr"), indices)
            offset += len(atom_chars)
            if match.group("modifier"):
                indices.append(modifier_chars.index(match.group("modifier"))+offset)
            offset += len(modifier_chars)
            if match.group("isotopes"):
                indices.append(isotopes.index(match.group("isotopes"))+offset)
            offset += len(isotopes)
            for i in range(0, 2):
                if match.group("multiplicity_" + str(i)):
                    indices.append(int(match.group("multiplicity_"+str(i)))+offset+mults*i)
        else:
            raise Exception("Could not encode atom", x)
    else:
        offset += len(atom_chars) + len(modifier_chars) + len(isotopes) + 2*mults
        if t == TokenType.BOND_TYPE or t == TokenType.EZSTEREO:
            indices.append(bond_chars.index(x))

        else:
            offset += len(bond_chars)
            if t == TokenType.RING_NUM:
                indices.append(x + offset - 1)
            else:
                offset += 36
                if t == TokenType.BRANCH_START or t == TokenType.BRANCH_END:
                    indices.append(branch_chars.index(x))
    if not indices:
        raise Exception("Could not encode", x)
    for index in indices:
        v[index]=True
    return v
