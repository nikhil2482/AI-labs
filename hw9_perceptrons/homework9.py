############################################################
# CIS 5210: Homework 9
############################################################

############################################################
# Imports
############################################################

# Include your imports here, if any are used.
import homework9_data as data

############################################################

student_name = ""

############################################################
# Section 1: Perceptrons
############################################################


class BinaryPerceptron(object):

    def __init__(self, examples, iterations):
        self.weights = {}
        for _ in range(iterations):
            for x, y in examples:
                prediction = self.predict(x)
                if prediction != y:
                    sign = 1 if y else -1
                    for key, value in x.items():
                        self.weights[key] = (
                            self.weights.get(key, 0) + sign * value
                        )

    def predict(self, x):
        score = sum(self.weights.get(key, 0) * value
                    for key, value in x.items())
        return score > 0


class MulticlassPerceptron(object):

    def __init__(self, examples, iterations):
        self.labels = []
        seen = set()
        for x, y in examples:
            if y not in seen:
                seen.add(y)
                self.labels.append(y)
        self.weights = {label: {} for label in self.labels}
        for _ in range(iterations):
            for x, y in examples:
                prediction = self.predict(x)
                if prediction != y:
                    correct_weights = self.weights[y]
                    wrong_weights = self.weights[prediction]
                    for key, value in x.items():
                        correct_weights[key] = (
                            correct_weights.get(key, 0) + value
                        )
                        wrong_weights[key] = (
                            wrong_weights.get(key, 0) - value
                        )

    def predict(self, x):
        best_label = None
        best_score = None
        for label in self.labels:
            weights = self.weights[label]
            score = sum(weights.get(key, 0) * value
                        for key, value in x.items())
            if best_score is None or score > best_score:
                best_score = score
                best_label = label
        return best_label


############################################################
# Section 2: Applications
############################################################


class IrisClassifier(object):

    def __init__(self, data):
        examples = [(self._features(x), y) for x, y in data]
        self.classifier = MulticlassPerceptron(examples, 20)

    def _features(self, instance):
        return {index: value for index, value in enumerate(instance)}

    def classify(self, instance):
        return self.classifier.predict(self._features(instance))


class DigitClassifier(object):

    def __init__(self, data):
        examples = [(self._features(x), y) for x, y in data]
        self.classifier = MulticlassPerceptron(examples, 40)

    def _features(self, instance):
        features = {index: value for index, value in enumerate(instance)}
        features["bias"] = 1
        return features

    def classify(self, instance):
        return self.classifier.predict(self._features(instance))


class BiasClassifier(object):

    def __init__(self, data):
        examples = [(self._features(x), y) for x, y in data]
        self.classifier = BinaryPerceptron(examples, 50)

    def _features(self, x):
        return {"x": x, "bias": 1}

    def classify(self, instance):
        return self.classifier.predict(self._features(instance))


class MysteryClassifier1(object):

    def __init__(self, data):
        examples = [(self._features(x), y) for x, y in data]
        self.classifier = BinaryPerceptron(examples, 50)

    def _features(self, instance):
        x1, x2 = instance
        return {
            "x1": x1,
            "x2": x2,
            "x1^2": x1 ** 2,
            "x2^2": x2 ** 2,
            "x1*x2": x1 * x2,
            "bias": 1,
        }

    def classify(self, instance):
        return self.classifier.predict(self._features(instance))


class MysteryClassifier2(object):

    def __init__(self, data):
        examples = [(self._features(x), y) for x, y in data]
        self.classifier = BinaryPerceptron(examples, 200)

    def _features(self, instance):
        x1, x2, x3 = instance
        return {
            "x1*x2*x3": x1 * x2 * x3,
        }

    def classify(self, instance):
        return self.classifier.predict(self._features(instance))


############################################################
# Section 3: Feedback
############################################################

# Just an approximation is fine.
feedback_question_1 = """
5
"""

feedback_question_2 = """
The most difficult part of this assignment was figuring out the
feature transformations for the two mystery data sets. Since I could
not immediately tell what geometric pattern separated the two
classes, I had to reason carefully from the given input and output
examples before settling on features such as squared terms and
products of coordinates. Getting the multiclass perceptron's
tie-breaking behavior correct when several classes scored equally
was also less obvious than the binary case.
"""

feedback_question_3 = """
I liked how the assignment built up from a simple binary perceptron
to a more general multiclass version, since it made the underlying
algorithm easier to internalize before applying it to real data sets
like Iris and the handwritten digits. Turning an abstract classifier
into something that recognizes actual images and flower
measurements was rewarding. If I could change one thing, I would
have liked a suggested starting point for visualizing the mystery
data sets, since a lot of the challenge was in guessing the right
transformation rather than in the perceptron algorithm itself.
"""

############################################################
# Section 4: Acknowledgements & reflections
############################################################

acknowledgements = """
I used Claude to help me when I was stuck with this homework
not by asking for the direct answer but ask me questions to
prompt me to develop and write the answer on my own. Using an AI
tool sped up my thinking process but I practiced tracing
through edge cases to make sure I understood what each function does.

I made sure to read through the resulting code and
reasoning so that I could still explain how each perceptron update
rule works and why each feature transformation was chosen, rather
than treating the file as a black box. This affected my workflow in
that most of my own time went into understanding the perceptron
algorithm and the mystery data sets' structure well enough to judge
whether the generated solution was reasonable, rather than into
typing out the code myself.
"""
