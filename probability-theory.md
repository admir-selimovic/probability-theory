
# Probability Theory

Probability theory offers means of quantifying uncertainties. It was originally developed to analyze the frequencies of repeatable events.
<br>
There is a variety of competing interpretations of probability. The dominant two streams are the frequentist and the Bayesian. When an outcome has a probability $P$ of occurring, it implies a repetition of the experiment infinitely many times and a ratio $p$ of the repetitions resulting in that outcome. Under frequentist inference, a hypothesis is tested without being assigned a probability. Probability is defined in terms of frequencies of the occurrence of events, or by relative proportions in populations. Frequency interpretations are defined by a ratio from an infinite series of trials.Such a probability is referred to as frequentist. However, probability can be thought of a degree of belief, in which case it is referred to as Bayesian.
<br><br>
In probability theory, one typically knows the parameters of the probability distribution while trying to predict properties of the samples, i.e., probability distribution is given and the probability of a specific event is to be determined.
<br>
The question of probability theory is: Given a data generating process, what are the properties of the outcomes?

## Kolmogorov Axioms

The roots of probability reach centuries into the past, but it is with Kolmogorov that probability theory becomes rigorous, i.e., it becomes a branch of mathematics. An axiomatic definition of probability is elaborated in his 1933 work *Foundations of Theory of Probability*.

### Axioms

#### Axiom 1

$$
P(E) \geq 0
$$

The probability of an event $P(E)$ is a non-negative real number not greater than 1.

#### Axiom 2

$$
P(S)=1
$$

The probability of the entire sample space $S$ equals unity.

#### Axiom 3

$$
P\left(\bigcup_{i=1}^{\infty} E_i\right)=\sum_{i=1}^{\infty} P\left(E_i\right.
$$

For any countable sequence of mutually exclusive events (disjoint sets) $E_1, E_2,...$, the probability of at least one of these events occurring is the sum of their respective probabilities.

### Deductions

##### The probability of the empty set

$$
P(\emptyset)=0
$$

The probability of an empty set is 0.

##### The complement rule

$$
P(\neg A)=1-P(A)
$$

The complement of an event is the event not occurring. Given the probability of an event, the probability of its complement is obtained by subtracting the given probability from 1.

##### Monotonicity

$$
\text { if } A \subseteq B \text { follows } P(A) \leq P(B)
$$

If A is a subset of, or equal to B, then the probability of A is less than or equal to the probability of B.

##### The sum rule, or the addition law of probability

$$
P(A \cup B)=P(A)+P(B)-P(A \cap B)
$$

The probability that the event $A$ or $B$ occurs is the probability of the union of $A$ and $B$,  $P(A ∪ B)$. The probability that both events $A$ and $B$ occur is the probability of the intersection of $A$ and $B$, $P(A ∩ B)$

## Random Variables

A random variable is one that can take on possible values/states randomly. One differentiates between discrete and continuous variables.

## Probability Distributions

A probability distribution is a function that describes the likelihood of a random variable assuming any of the possible values, i.e., the likelihood of an event or outcome. There is a principle disambiguation between two kinds of probability distributions (in accordance with the nature of variables): probability mass function (PMF) for discrete variables, and probability density function (PDF) for continuous variables.

### Probability Mass Function (PMF)

The probability mass function (PMF) maps from a state of a random variable to the probability of that random variable taking on that state. It operates on **discrete** variables.

$$
f(x)=\frac{d P(x)}{d x}
$$

where P is the probability measure; the likelihood that random variable takes a specific value of x.

PMF satisfies the following properties:

$$
\forall x \in X, 0 \leq P(x) \leq 1
$$

Probability of an event must be a value between 0 (impossible) and 1 (certain).

$$
\sum_{x \in X} P(x)=1
$$

The sum of all probabilities is 1. The distribution is normalised.

##### Example

If the PMF of rolling a die is computed, where random variable $X = \{1, 2, 3, 4, 5, 6\}$, PMF($X$) draws  as follows:

PMF can act on more than one variable. Such a probability distribution is referred to as **joint probability distribution**.

### Probability Density Function (PDF)

Probability density function (PDF) operates on **continuous** random variables. PDF for any random variable $X$ is given as follows:

$$
f(x)=\frac{d P(x)}{d x}
$$

PDF satisfies the following **properties**:

$$
\forall x \in X, P(x) \geq 0
$$

Notice that, in difference to PMF, the probability does not need to be smaller or equal to 1.

$$
\int f(x) d x=1
$$

PDF integrates to unity.

PDF gives a geometrical interpretation of the probability of an event. In difference to PMF, it does not output the probability of any state, but the probability of landing inside an infinitesimal region with volume 𝛿x is given by p(x)𝛿x.


### Cumulative Distribution Function (CDF)

Cumulative Distribution Function (CDF) for any random variable $X$ is the probability that $X$ assumes a value less than or equal to $x$:

$$
F(x)=P(X \leq x)
$$

Given a discrete random variable (and the associated probability mass function), the definition of CDF can be reformulated as follows:

$$
F(x)=\sum_{k=1}^n P\left(X=x_k\right)
$$

where $x_n$ is the largest possible value of $X$ that is less than or equal to $x$.

CDF has the following **properties**:

$$
\forall x \in X, 0 \leq F(x) \leq 1
$$

For all $x$ in $X$, $F(x)$ is greater than or equal to 0 and less than or equal to 1.

$$
\lim _{x \rightarrow-\infty} F(-\infty)=0, \lim _{x \rightarrow+\infty} F(\infty)=1
$$

As x approaches negative infinity, $F(x)$ approaches (or equals) 0. As $x$ approaches positive infinity, $F(x)$ approaches (or equals) 1.

$$
\forall b \geq a, F(b) \geq F(a)
$$

$F(x)$ is a non-decreasing function of $x$.



## Joint, Marginal, and Conditional Probabilities

### Marginal Probability

Marginal probability is the one of an event occuring, $P(A)$, irrespective of any other variable.

### Joint Probability

Joint probability is the one of two events, $P(A,B)$ occurring simultaneously.

### Conditional Probability

Conditional probability is the one of an event $A$ occuring in the presence of another event $B$:

$$
P(A|B) = \frac{P(A, B)}{P(B)}, \quad P(B) \neq 0
$$
<br><br>
$$
\text{Conditional Probability} = \frac{\text{Joint Probability}}{\text{Marginal Probability}}
$$


## Expectation, Variance, and Covariance

### Expectation

Expectation, or expected value, or mean of a discrete random variable $X$ having a probability mass function $f(x)$ is defined by

$$
E[X]=\sum_{i=1}^n x_i f\left(x_i\right)
$$

Expected value is a prediction for the outcome of an experiment. It is most useful when the outcome is not likely to deviate much from the expected value.
Expected value of a random variable $X$ is its theoretical mean. It is based on the distribution of the random variable $X$.  It is a weighted average of the possible values that $X$ can assume; all the possible values of $X$ are weighted according to the probability $f(x)$ that $X$ assumes it.

Expectation value can be calculated also for functions of the random variable.
<br><br>
\[
Suppose that we are given a discrete random variable along with its probability mass function and that we want to compute the expected value of some function of X, say, g(X). How can we accomplish this? One way is as follows: Since g(X) is itself a discrete random variable, it has a probability mass function, which can be determined from the probability mass function of X. Once we have determined the probability mass function of g(X), we can compute E[g(X)] by using the definition of expected value.
\]

$$
E[g(X)]=\sum_{i=1}^n g\left(x_i\right) f\left(x_i\right)
$$

Expected value of a **continuous random variable** $X$ is given by

$$
E[X]=\int_{-\infty}^{\infty} x f(x) d x
$$

The equation is the continuous analog of the  expected value of a discrete random variable, where the summing over all possible values is substituted by integration.
<br><br>
Expectation  is often  denoted by μ.

### Variance

Expected value $E[X]$ yields the weighted average of the possible values of $X$. Variance measures the deviation, or spread of those values. It is given by the averaged squared distance between $X$ and its mean. If $X$ is a random variable with mean μ, then the variance of $X$ is defined by

$$
\text{Var}(X)=E\left[(X-\mu)^2\right]=\sum_{i=1}^n\left(x_i-\mu\right)^2 f\left(x_i\right)
$$

An alternative formula for variance $Var(X)$, which offers the easiest way to compute it, is as follows:

$$
\text{Var}(X)=E\left[X^2\right]-(E[X])^2
$$

Variance is often denoted by $σ^2$.

### Covariance

Data sets are often of more than one dimension. The aim of statistical analysis of these data sets is usually to determine relationships between its dimensions. Covariance is a measure of how much the dimensions vary from the mean with respect to each other. It is measured always among two dimensions. If there are more than two dimensions in the data set, there is more than one covariance measurement that can be calculated.

For random variables $X$ and $Y$, the covariance is defined as the expected value of the product of their deviations from their individual expected values:

$$
\text{Cov}(X, Y)=E[(X-E[X])(Y-E[Y])]
$$

In case the random variable pair $(X, Y)$ assumes $n$ possible values $(x_i, y_i)$ with **equal probabilities** $p_i = 1/n, Cov(X,Y)$ can be expressed as follows:

$$
\text{Cov}(X, Y)=\frac{1}{n} \sum_{i=1}^n\left(x_i-E[X]\right)\left(y_i-E[Y]\right)
$$

Otherwise, if the random variable pair $(X, Y)$ assumes $n$ possible values $(x_i, y_i)$ with **unequal probabilities** $p_i$, the covariance can be expressed as follows:

$$
\text{Cov}(X, Y)=\sum_{i=1}^n p_i\left(x_i-E[X]\right)\left(y_i-E[Y]\right)
$$

Covariance is often denoted by $σ^2(X,Y)$.

## Common Probability Distributions

The probability of outcomes for discrete random variables can be summarized using discrete probability distributions.

For outcomes that can be ordered, the probability of an event equal to or less than a given value is defined by the cumulative distribution function, or CDF for short. The inverse of the CDF is called the percentage-point function and will give the discrete outcome that is less than or equal to a probability.

PMF: Probability Mass Function, returns the probability of a given outcome.
CDF: Cumulative Distribution Function, returns the probability of a value less than or equal to a given outcome.
PPF: Percent-Point Function, returns a discrete value that is less than or equal to the given probability.

### Bernoulli  Distribution (Discrete)

**A single binary outcome** has a Bernoulli distribution. Any event of one trial and two possible and mutually exclusive outcomes follows a Bernoulli distribution. Random variable X assumes the value of 1 with probability $p$ if a success occurs, and 0 with probability $1-p$ if a failure occurs.
<br><br>
The probability mass function (PMF) of the Bernoulli distribution is given as

$$
P(X=x)=p^x(1-p)^{1-x}
$$

The probability that the random variable assumes the value of 0:

$$
P(X=0)=p^0(1-p)^{1-0}=1-p
$$

The probability that the random variable assumes the value of 1:

$$
P(X=1)=p^1(1-p)^{1-1}=p
$$

**Mean/expectation**:

$$
\mu=p
$$

**Variance**:

$$
\sigma^2=p(1-p)
$$

Some other common discrete probability distributions are built on the assumptions of independent Bernoulli trails. Among those are binomial, geometric, and negative binomial distributions.

### Binomial  Distribution (Discrete)

**A sequence of binary outcomes** has a Binomial distribution.
Each of *n* independent trials can result in one of two possible outcomes, labelled $success$ (assumed with probability $p$) and $failure$ (assumed with probability $1-p$). The random variable $X$ represents the number of successes in $n$ trials, i.e., the number of successes in $n$ independent Bernoulli trials has a binomial distribution.
<br><br>
The independent variable $X$ has a binomial distribution given by the probability mass function (PMF)

$$
P(X=x)=\left(\begin{array}{l}
n \\
x
\end{array}\right) p^x(1-p)^{n-x}
$$

for $x = 0,...,n$.

**Combinations formula**, or the binomial coefficient is defined as

$$
\left(\begin{array}{l}
n \\
x
\end{array}\right)=\frac{n !}{x !(n-x) !}
$$

where $n$ represents the total number of items and $x$ the number of items chosen at a time.

**Mean/expectation**:

$$
\mu=n p
$$

**Variance**:

$$
\sigma^2=p(1-p)
$$

___
##### Example:

A die is rolled $n = 50$ times. What is the probability a 3 comes up five times?

Let the random variable $X$ represent the number of 3's in $n = 50$ rolls. $X$ is a binomially distributed random variable with total number of trials $n = 50$ and the probability of each trial yielding a successful resul $p = 1/6$, often written as $X$~$B(50, 1/6)$.

The probability that the random variable X takes on the value 5 is
$P(X=5)= {50 \choose 5} {(\frac{1}{6})}^5 {(1 - \frac{1}{6})}^{50-5} = 0.1118$



### Multinoulli/Categorical Distribution (Discrete)

**A single categorical outcome** has a Multinoulli distribution.
The multinoulli/categorical distribution is a special case of the multinomial distribution and a generalisation of the Bernoulli distribution. It is a distribution over a single discrete rrandom variable $X$ with $k$ different states.

Multinoulli distributions are often used to refer to distributions over categories of objects. Therefore, it is not usually assumed that, for instance, state 1 assumes the numerical value 1. Due to this, the expectation and variance of random variables with a multinoulli distribution is usually omitted.

##### Example:

A single roll of a die whose outcome will be in $S = \{1,2,3,4,5,6\}$, $k=6$

##### Example:

In machine learning, a common case of a Multinoulli distribution is a multi-class classification of a single example into one of $k$ classes.

### Multinomial Distribution (Multivariate (joint))

**A sequence of categorical outcomes** has a Multinomial distribution. The repetition of multiple independent Multinoulli trials follows a multinomial distribution.
<br>
The multinomial distribution is a generalisation of the binomial distribution. In the binomial distribution, there are only two possible outcomes in any one individual trial, labeled $success$ and $failure$. In the multinomial distribution, the number of possible outcomes can be greater than two.
<br><br>
Suppose that there are $n$ independent trials and each one results in one of $k$ mutually exclusive outcomes. In any single trial, the $k$ outcomes occur with probabilities $p_1,...,p_k$, which sum to 1 and stay constant from trial to trial.
<br><br>
The random variable $X$ takes on values $0,...,n$ and the distribution is given by

$$
P\left(X_1=x_1, \ldots, X_k=x_k\right)=\frac{n !}{x_{1} ! \ldots x_{k} !} p_1^{x_1} \ldots p_k^{x_k}
$$

**Mean/Expectation**:

$$
\mu\left(X_i\right)=n p_i
$$

**Variance**:

$$
\sigma^2\left(X_i\right)=n p_i\left(1-p_i\right)
$$

##### Example:

A bag contains 14 red, 9 green, and 7 blue marbles. 10 marbles are chosen at random with replacement (the probabilities stay constant from trial to trial). What is the probability that three are red, one is green, and six are blue?

$$
P(x_1=3, x_2=1,x_3=6) = \frac{10!}{3! 1! 6!} ({\frac{14}{30}})^3 ({\frac{9}{30}})^1 ({\frac{7}{30}})^6 = 0.0041
$$


### Normal/Gaussian Distribution (Continuous)

Normal/Gaussian distribution is a type of continuous probability distribution over a real-valued random variable. Its probability density function (PDF) is given by

$$
\mathcal{N}\left(x ; \mu, \sigma^2\right)=\sqrt{\frac{1}{2 \pi \sigma^2}} \exp \left(-\frac{1}{2 \sigma^2}(x-\mu)^2\right)
$$

Normal distribution is controlled by two parameters. $\mu$ is the mean or expectation of the distribution which gives the coordinate of its bell curve's central peak; and $\sigma$ is its standard deviation, while the variance is denoted by $\sigma^2$.
<br><br>
Although we will not make direct use of this formula, it is interesting to note that it involves two of the famous constants of mathematics: $π$ (the area of a circle of radius 1) and $e$ (which is the base of the natural logarithms). Also note that this formula is completely specified by the mean value μ and the standard deviation $σ$.


### Exponential Distribution (Continuous)

The probability density function (PDF) of an exponential distribution:

$$
f(x ; \lambda)= \begin{cases}\lambda \exp (-\lambda x) & x \geq 0 \\ 0 & x<0\end{cases}
$$

where λ is the rate parameter.
<br><br>
The exponential distribution assigns probability of zero to all negative values of x. It places a sharp point at $x=0$.

### Laplace Distribution (continuous)

$$
f(x ; \mu, b)=\frac{1}{2 b} \exp \left(-\frac{|x-\mu|}{b}\right)
$$

### Geometric Distribution (discrete)

$n$ independent trials, each with a probability $p, 0<p<1$, of resulting in success, are performed until a success occurs. The discrete random variable $X$ equals the nmber of trials $n$.
<br>
Probability mass function (PMF) of the geometric distribution is defined as

$$
P(X=n)=(1-p)^{n-1} p
$$

where $n = 1,2,...$
<br><br>
Any random variable $X$ whose PMF is given by the equation for the PMF of the geometric distribution is said to be a geometric random variable with parameter $p$.

### Poisson Distribution

A discrete random variable $X$ takes on one of the values 0,1,2,...

$$
f(n ; \lambda)=P(X=n)=\frac{\lambda^n e^{-\lambda}}{n !}
$$

where $e$ is Euler's number and λ is the rate parameter, a positive real number that is equal to the expected value of $X$ and also its variance.

## Bayes' Theorem

In difference to the frequentist interpretation, Bayesian probability is not related directly to the rate at which events occur, but to qualitative levels of certainty.
<br>
In the Bayesian interpretation of probability, instead of the frequency of a phenomenon, probability representrs a reasonable expectation, or the current state of knowledge, or a quantification of a personal belief which is updated with new data. It describes the probability of an event, based on prior knowledge of conditions that might be related to the event.
<br>
In the Bayesian view, a probability is assigned to a hypothesis. Bayesian probability belongs to the category of evidential probabilities; to evaluate the probability of a hypothesis, the Bayesian probabilist specifies some prior probability, which is then updated to a posterior probability in the light of new, relevant data (evidence).


Bayes' theorem is given as

$$
P(A \mid B)=\frac{P(B \mid A) P(A)}{P(B)}
$$

where $P(A|B)$ is the probability of event $A$ occurring, given that $B$ is true (conditional probability); $P(A)$ and $P(B)$ are the probabilities of an independent occurrence (marginal probability).

##### Example:

**(1)** 10% of all days this month are rainy; **(2)** 40% of all days start cloudy; **(3)** 50% of all rainy days start cloudy.
What is the chance of rain, given that today started cloudy?

**(1)** $P(\text{Rain}) = 0.1$
<br>
**(2)** $P(\text{Cloud}) = 0.4$
<br>
**(3)** $P(\text{Cloud|Rain}) = 0.5$
<br><br>
$$
P(\text{Rain|Cloud}) = \frac{P(\text{Cloud|Rain}) \times P(\text{Rain})}{P(\text{Cloud})} = \frac{0.5 \times 0.1}{0.4} = 0.125 \text{ or } 12.5\%
$$


---
