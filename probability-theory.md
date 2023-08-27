# Probability Theory

Probability theory offers a framework for quantifying uncertainties. It was initially developed to analyse the frequencies of repeatable events.

There are various competing interpretations of probability. The two dominant streams are frequentist and Bayesian. When an outcome has a probability denoted as $P$ of occurring, it implies an experiment's repetition an infinite number of times, with a ratio $p$ of those repetitions resulting in that specific outcome. Frequentist inference tests hypotheses without assigning them probabilities. Probability, from this viewpoint, is defined based on event frequencies or relative proportions within populations. Interpretations based on frequency are known as frequentist. However, probability can also be understood as a degree of belief, leading to the Bayesian perspective.

In probability theory, the usual scenario involves knowing the parameters of the probability distribution while attempting to predict properties of the samples. This means the probability distribution is given, and the task is to determine the probability of a specific event occurring.

The fundamental question of probability theory is: Given a data-generating process, what can we infer about the properties of the outcomes?

## Kolmogorov Axioms

The origins of probability theory reach back centuries, but it was with Kolmogorov that the field gained rigorous mathematical grounding. His 1933 work, *Foundations of the Theory of Probability*, provided an axiomatic definition of probability.

### Axioms

#### Axiom 1

$$
P(E) \geq 0
$$

The probability of an event $E$, denoted as $P(E)$, is a non-negative real number that does not exceed 1.

#### Axiom 2

$$
P(S)=1
$$

The probability of the entire sample space $S$ is equal to 1.

#### Axiom 3

$$
P\left(\bigcup_{i=1}^{\infty} E_i\right)=\sum_{i=1}^{\infty} P\left(E_i\right)
$$

For any countable sequence of mutually exclusive events (disjoint sets) $E_1, E_2, \ldots$, the probability of at least one of these events occurring is the sum of their respective probabilities.

### Deductions

##### Probability of the Empty Set

$$
P(\emptyset)=0
$$

The probability of an empty set is 0.

##### Complement Rule

$$
P(\neg A)=1-P(A)
$$

The complement of an event is the event not occurring. Given the probability of an event, the probability of its complement is obtained by subtracting the given probability from 1.

##### Monotonicity

$$
\text{if } A \subseteq B \text{, then } P(A) \leq P(B)
$$

If event $A$ is a subset of, or equal to event $B$, then the probability of event $A$ is less than or equal to the probability of event $B$.

##### Sum Rule, or Addition Law of Probability

$$
P(A \cup B)=P(A)+P(B)-P(A \cap B)
$$

The probability that event $A$ or event $B$ occurs is the probability of their union, $P(A \cup B)$. The probability that both events $A$ and $B$ occur is the probability of their intersection, $P(A \cap B)$.

## Random Variables

A random variable is a variable that can take on various possible values or states in a random manner. We distinguish between discrete and continuous variables.

## Probability Distributions

A probability distribution is a function that describes the likelihood of a random variable assuming different possible values—essentially, the likelihood of a particular event or outcome occurring. Two main types of probability distributions exist, based on the nature of the variables: the probability mass function (PMF) for discrete variables and the probability density function (PDF) for continuous variables.

### Probability Mass Function (PMF)

The probability mass function (PMF) maps the state of a random variable to the probability of that variable taking on that particular state. It is used for **discrete** variables.

$$
f(x)=\frac{d P(x)}{d x}
$$

Here, $P$ represents the probability measure, denoting the likelihood that the random variable takes on the specific value $x$.

The PMF adheres to the following properties:

$$
\forall x \in X, 0 \leq P(x) \leq 1
$$

The probability of an event must lie between 0 (impossible) and 1 (certain).

$$
\sum_{x \in X} P(x)=1
$$

The sum of all probabilities is 1, ensuring the distribution is normalised.

The PMF can operate on more than one variable. In such cases, it is referred to as a **joint probability distribution**.

### Probability Density Function (PDF)

The probability density function (PDF) is used for **continuous** random variables. For any random variable $X$, the PDF is expressed as follows:

$$
f(x)=\frac{d P(x)}{d x}
$$

The PDF complies with the following **properties**:

$$
\forall x \in X, P(x) \geq 0
$$

Unlike the PMF, the probability in the PDF does not need to be smaller than or equal to 1.

$$
\int f(x) d x=1
$$

The integral of the PDF over its range is equal to 1, providing normalisation.

The PDF offers a geometric interpretation of the probability of an event. Unlike the PMF, it doesn't provide the probability of specific states. Instead, it offers the probability of landing within an infinitesimal region with volume $\delta x$, given by $p(x) \delta x$.

### Cumulative Distribution Function (CDF)

The Cumulative Distribution Function (CDF) for a random variable $X$ gives the probability that $X$ assumes a value less than or equal to $x$:

$$
F(x)=P(X \leq x)
$$

For discrete random variables (with the associated probability mass function), the CDF can be expressed as follows:

$$
F(x)=\sum_{k=1}^n P\left(X=x_k\right)
$$

Here, $x_n$ represents the largest possible value of $X$ that is less than or equal to $x$.

The CDF possesses the following **properties**:

$$
\forall x \in X, 0 \leq F(x) \leq 1
$$

For any $x$ in the range of $X$, the CDF value is between 0 and 1.

$$
\lim _{x \rightarrow-\infty} F(-\infty)=0, \lim _{x \rightarrow+\infty} F(\infty)=1
$$

As $x$ approaches negative infinity, $F(x)$ approaches (or equals) 0. As $x$ approaches positive infinity, $F(x)$ approaches (or equals) 1.

$$
\forall b \geq a, F(b) \geq F(a)
$$

The function $F(x)$ is non-decreasing with respect to $x$.

## Joint, Marginal, and Conditional Probabilities

### Marginal Probability

Marginal probability, denoted as $P(A)$, represents the probability of an event occurring independently of any other variable.

### Joint Probability

Joint probability, $P(A,B)$, quantifies the simultaneous occurrence of two events, $A$ and $B$.

### Conditional Probability

Conditional probability, denoted as $P(A|B)$, describes the probability of event $A$ occurring in the presence of another event $B$:

$$
P(A|B) = \frac{P(A, B)}{P(B)}, \quad P(B) \neq 0
$$

Alternatively,

$$
\text{Conditional Probability} = \frac{\text{Joint Probability}}{\text{Marginal Probability}}
$$


## Expectation, Variance, and Covariance

### Expectation

Expectation, also known as the expected value or mean, of a discrete random variable $X$ with a probability mass function $f(x)$ is defined by:

$$
E[X] = \sum_{i=1}^n x_i f(x_i)
$$

The expected value is a prediction for the outcome of an experiment. It is particularly useful when the outcome is not expected to deviate significantly from the expected value. The expected value of a random variable $X$ represents its theoretical mean, based on the distribution of $X$. It is computed as a weighted average of all possible values of $X$, where each value is weighted by the probability $f(x)$ associated with it.

The expectation can also be calculated for functions of the random variable. For a function $g(X)$, the expectation is given by:

$$
E[g(X)] = \sum_{i=1}^n g(x_i) f(x_i)
$$

For continuous random variables, the expectation is calculated through integration:

$$
E[X] = \int_{-\infty}^{\infty} x f(x) \, dx
$$

Expectation is often denoted by the symbol μ.

### Variance

The expected value $E[X]$ provides a weighted average of the possible values of $X$. Variance measures the spread or deviation of these values from the mean. For a random variable $X$ with mean μ, the variance is defined as:

$$
\text{Var}(X) = E[(X - \mu)^2] = \sum_{i=1}^n (x_i - \mu)^2 f(x_i)
$$

An alternative formula for variance is:

$$
\text{Var}(X) = E[X^2] - (E[X])^2
$$

Variance is often represented by the symbol $σ^2$.

### Covariance

Data sets often have multiple dimensions, and statistical analysis aims to understand relationships between these dimensions. Covariance measures the extent to which dimensions vary together. It's a measure of the joint variability between two random variables $X$ and $Y$. The covariance between $X$ and $Y$ is defined as:

$$
\text{Cov}(X, Y) = E[(X - E[X])(Y - E[Y])]
$$

In the case where $(X, Y)$ takes on $n$ possible values $(x_i, y_i)$ with equal probabilities $p_i = 1/n$, the covariance can be calculated as:

$$
\text{Cov}(X, Y) = \frac{1}{n} \sum_{i=1}^n (x_i - E[X])(y_i - E[Y])
$$

For unequal probabilities $p_i$, the formula becomes:

$$
\text{Cov}(X, Y) = \sum_{i=1}^n p_i (x_i - E[X])(y_i - E[Y])
$$

Covariance is often denoted by $σ^2(X, Y)$.

## Common Probability Distributions

Discrete random variables are summarized using discrete probability distributions.

For ordered outcomes, the cumulative distribution function (CDF) provides the probability of an event less than or equal to a value. The inverse of the CDF is the percentage-point function (PPF), which gives the outcome corresponding to a probability.

- PMF: Probability Mass Function, gives the probability of an outcome.
- CDF: Cumulative Distribution Function, provides the probability of a value up to a given outcome.
- PPF: Percent-Point Function, yields a value less than or equal to a probability.

### Bernoulli Distribution (Discrete)

A Bernoulli distribution models a binary outcome. In a single trial with two mutually exclusive outcomes, the random variable $X$ takes value 1 with probability $p$ (success) and value 0 with probability $1-p$ (failure).

The PMF of the Bernoulli distribution is:

$$
P(X=x) = p^x (1-p)^{1-x}
$$

Mean: $\mu = p$.
Variance: $\sigma^2 = p(1-p)$.

### Binomial Distribution (Discrete)

The binomial distribution models a sequence of binary outcomes. In $n$ independent trials, each with probability $p$ of success, the random variable $X$ represents the number of successes. It's a generalization of the Bernoulli distribution.

The PMF of the binomial distribution is:

$$
P(X=x) = \binom{n}{x} p^x (1-p)^{n-x}
$$

where $\binom{n}{x}$ is the binomial coefficient.

Mean: $\mu = np$.
Variance: $\sigma^2 = np(1-p)$.

### Multinoulli / Categorical Distribution (Discrete)

The multinoulli distribution is a categorical distribution over $k$ states. It's a generalization of the Bernoulli distribution where there are more than two possible outcomes.

### Multinomial Distribution (Multivariate, Joint) - Discrete

The multinomial distribution models sequences of categorical outcomes. It generalizes the binomial distribution to more than two outcomes. In $n$ independent trials with $k$ mutually exclusive outcomes, the random variables $X_i$ represent the counts of each outcome.

Mean of $X_i$: $\mu(X_i) = np_i$.
Variance of $X_i$: $\sigma^2(X_i) = np_i(1-p_i)$.

### Normal/Gaussian Distribution (Continuous)

The normal distribution is a continuous probability distribution over real-valued variables. It's described by its mean $\mu$ and standard deviation $\sigma$.

The PDF of the normal distribution is:

$$
\mathcal{N}(x; \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

### Exponential Distribution (Continuous)

The exponential distribution models the time between events in a Poisson process. Its PDF is:

$$
f(x; \lambda) = \begin{cases} \lambda \exp(-\lambda x) & x \geq 0 \\ 0 & x < 0 \end{cases}
$$

### Laplace Distribution (Continuous)

The Laplace distribution models continuous variables with sharp peaks at the mean and tails that decay exponentially. Its PDF is:

$$
f(x; \mu, b) = \frac{1}{2b} \exp\left(-\frac{|x - \mu|}{b}\right)
$$

### Geometric Distribution (Discrete)

The geometric distribution models the number of trials needed for a success in a sequence of independent Bernoulli trials. The PMF is:

$$
P(X=n) = (1-p)^{n-1} p
$$

Mean: $\mu = \frac{1}{p}$.
Variance: $\sigma^2 = \frac{1-p}{p^2}$.

### Poisson Distribution (Discrete)

The Poisson distribution models the number of events occurring in a fixed interval of time or space. The PMF is:

$$
P(n; \lambda) = \frac{\lambda^n e^{-\lambda}}{n!}
$$

where $\lambda$ is the rate parameter.

## Bayes' Theorem

In the Bayesian interpretation of probability, it represents a level of certainty or a quantification of belief, which is updated with new data. Bayes' theorem relates the probability of an event based on new evidence to the prior probability of that event.

Bayes' theorem is given by:

$$
P(A | B) = \frac{P(B | A) P(A)}{P(B)}
$$

Here, $P(A | B)$ is the probability of event $A$ given that event $B$ has occurred. $P(B | A)$ is the probability of event $B$ given that event $A$ has occurred. $P(A)$ and $P(B)$ are the probabilities of events $A$ and $B$ respectively.

This theorem is central to Bayesian statistics and allows for updating probabilities as new information becomes available. It is widely used in various fields including machine learning, decision theory, and more.

