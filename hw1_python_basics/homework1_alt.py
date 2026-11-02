# import numpy
# import nltk

############################################################
# CIS 5210: Homework 1
############################################################

student_name = ""

############################################################
# Section 1: Python Concepts
############################################################

python_concepts_question_1 = """
Python is strongly typed, meaning every object has a fixed type and the
interpreter does not allow operations that are incompatible with that type.
For example, "foo" + 2 raises a TypeError because you cannot add a string
and an integer. Python will not silently coerce one into the other.

Python is also dynamically typed, meaning variables come into existence when
first assigned and their type is determined automatically at runtime. If you
assign the same variable again, type(var) is updated to reflect the new
value. For example:
    x = "foo"   # x is a str
    x = 2       # x is now an int -- no declaration needed, no error
Unlike Java (static typing), there are no type signatures on functions, and
type errors are only caught at runtime.
"""

python_concepts_question_2 = """
The problem is that lists are mutable and therefore not hashable, and Python
dictionary keys must be hashable. Using a list as a key raises a TypeError:
    "unhashable type: 'list'"

Every object has a fixed type (strong typing), and the dict type requires
keys whose type supports hashing -- lists do not.

The fix is to use tuples instead, because tuples are immutable and hashable:
    points_to_names = {(0, 0): "home", (1, 2): "school", (-1, 1): "market"}
"""

python_concepts_question_3 = """
concatenate1 creates N new strings for a list of N elements. Because strings
are immutable in Python, each += builds a brand-new string object that copies
all previously accumulated characters into it, so the total work is
proportional to 1 + 2 + ... + N = O(N^2) character copies.

concatenate2 calls join, which creates exactly one new string over its entire
execution. It makes a single pass and allocates the result once, running in
O(N) time.

concatenate2 is significantly faster for large inputs. For example, if you
have 10,000 strings of average length 10, concatenate1 does roughly 50
million character copies while concatenate2 does roughly 100,000.
"""

############################################################
# Section 2: Working with Lists
############################################################


def extract_and_apply(lst, p, f):
    return [f(x) for x in lst if p(x)]


def concatenate(seqs):
    return [item for seq in seqs for item in seq]


def transpose(matrix):
    return [[matrix[r][c] for r in range(len(matrix))]
            for c in range(len(matrix[0]))]


############################################################
# Section 3: Sequence Slicing
############################################################


def copy(seq):
    return seq[:]


def all_but_last(seq):
    return seq[:-1] if seq else seq[:]


def every_other(seq):
    return seq[::2]


############################################################
# Section 4: Combinatorial Algorithms
############################################################


def prefixes(seq):
    for i in range(len(seq) + 1):
        yield seq[:i]


def suffixes(seq):
    for i in range(len(seq) + 1):
        yield seq[i:]


def slices(seq):
    for start in range(len(seq)):
        for end in range(start + 1, len(seq) + 1):
            yield seq[start:end]


############################################################
# Section 5: Text Processing
############################################################


def normalize(text):
    return " ".join(text.lower().split())


def no_vowels(text):
    return "".join(c for c in text if c not in "aeiouAEIOU")


def digits_to_words(text):
    names = {
        "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
        "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
    }
    words = [names[c] for c in text if c.isdigit()]
    return " ".join(words)


def to_mixed_case(name):
    words = [w for w in name.split("_") if w]
    if not words:
        return ""
    return words[0].lower() + "".join(w.capitalize() for w in words[1:])


############################################################
# Section 6: Polynomials
############################################################


class Polynomial(object):

    def __init__(self, polynomial):
        self.polynomial = tuple(polynomial)

    def get_polynomial(self):
        return self.polynomial

    def __neg__(self):
        return Polynomial((-c, p) for c, p in self.polynomial)

    def __add__(self, other):
        return Polynomial(self.polynomial + other.polynomial)

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        return Polynomial(
            (c1 * c2, p1 + p2)
            for c1, p1 in self.polynomial
            for c2, p2 in other.polynomial
        )

    def __call__(self, x):
        return sum(c * x ** p for c, p in self.polynomial)

    def simplify(self):
        combined = {}
        for c, p in self.polynomial:
            combined[p] = combined.get(p, 0) + c
        terms = [(c, p) for p, c in combined.items() if c != 0]
        if not terms:
            terms = [(0, 0)]
        terms.sort(key=lambda cp: cp[1], reverse=True)
        self.polynomial = tuple(terms)

    def __str__(self):
        if not self.polynomial:
            return ""

        def term_str(c, p, first):
            if first:
                sign = "-" if c < 0 else ""
            else:
                sign = " - " if c < 0 else " + "
            abs_c = abs(c)
            if p == 0:
                return sign + str(abs_c)
            elif p == 1:
                if abs_c == 1:
                    return sign + "x"
                return sign + str(abs_c) + "x"
            else:
                if abs_c == 1:
                    return sign + "x^" + str(p)
                return sign + str(abs_c) + "x^" + str(p)

        parts = []
        for i, (c, p) in enumerate(self.polynomial):
            parts.append(term_str(c, p, i == 0))
        return "".join(parts)


############################################################
# Section 7: Python Packages
############################################################


def sort_array(list_of_matrices):
    import numpy as np
    combined = np.concatenate([m.flatten() for m in list_of_matrices])
    return np.sort(combined.astype(int))[::-1]


def POS_tag(sentence):
    import string
    import nltk
    tokens = nltk.word_tokenize(sentence.lower())
    stops = set(nltk.corpus.stopwords.words("english"))
    punct = set(string.punctuation)
    filtered = [t for t in tokens if t not in stops and t not in punct]
    return nltk.pos_tag(filtered)


#############################################################
# Section 8: Feedback
#############################################################

feedback_question_1 = """
5 hours.
"""

feedback_question_2 = """
Installing the NLTK package was a bit difficult for me, I'm not sure why
but ultimately I was able to install it. I have never used NLTK before
so this part of the homework took more time.
"""

feedback_question_3 = """
I liked the working wtih lists portion before because I'm familiar with
the concept already and this part of the homework took the least time.
"""

############################################################
# Section 9: Acknowledgements & reflections
############################################################

acknowledgements = """
I used Claude to help me when I was stuck with this homework
not by asking for the direct answer but ask me questions to
prompt me to develop and write the answer on my own. Using an AI
tool sped up my thinking process but I practiced tracing
through edge cases (e.g. Polynomial __str__ special cases, generator
boundaries) to make sure I understood what each function does.
"""
