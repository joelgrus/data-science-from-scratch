
num_variables = 100
num_points = 10000

data = [[random.random() for _ in range(num_variables)]
        for _ in range(num_points)]

def output(row):
    average = sum(row) / num_variables
    return 1 if average > 0.5 else 0

outcomes = map(output, data)

def predictor_using(i):
    def prediction(row):
        return 1 if row[i] > 0.5 else 0
    return prediction

weak_learners = map(predictor_using, range(num_variables))

def majority_vote(votes):
    c = Counter(votes)
    return c.most_common(1)[0][0]

def majority_predictor(row, predictors=weak_learners):
    return majority_vote(predictor(row) 
                         for predictor in predictors)

def majority_subpredictor(row, n):
    subpredictors = random.sample(weak_learners, n)
    return majority_vote(predictor(row) 
                         for predictor in subpredictors)


def classify(predictor):
    results = Counter()
    for x, y in zip(data, outcomes):
        prediction = predictor(x)
        if y and prediction:
            results["tp"] += 1
        elif y:
            results["fn"] += 1
        elif prediction:
            results["fp"] += 1
        else:
            results["tn"] += 1
    return results

def precision_and_recall(counts):
    precision = counts["tp"] / (counts["tp"] + counts["fp"])
    recall = counts["tp"] / (counts["tp"] + counts["fn"])
    return precision, recall

for i in range(num_variables):
    c = classify(predictor_using(i))
    precision, recall = precision_and_recall(c)
    print i, precision, recall

ensemble = classify(majority_predictor)
precision, recall = precision_and_recall(ensemble)
print "ensemble", precision, recall

for n in range(5,100):
    predictor = partial(majority_predictor,predictors=weak_learners[:n])
    ensemble = classify(predictor)
    precision, recall = precision_and_recall(ensemble)
    print n, precision, recall


def f(*args, **kwargs):
    print args
    print kwargs




def B(alpha, beta):
    return math.gamma(alpha + beta) / math.gamma(alpha) / math.gamma(beta)

def beta_pdf(x, alpha=1, beta=1):
    return x ** (alpha - 1) * (1 - x) ** (beta - 1) * B(alpha, beta)

xs = [i / 100 for i in range(1,100)]

alpha = .1
beta = .1
ys = [beta_pdf(x, alpha, beta) for x in xs]
plt.plot(xs, ys)
plt.show()

def choose(n, k):
    return math.factorial(n) // math.factorial(n - k) // math.factorial(k)


