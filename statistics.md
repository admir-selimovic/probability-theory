

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import scipy.stats as stats
from scipy.integrate import quad

pd.options.display.float_format = '{:<.4f}'.format

%matplotlib inline
```

# Statistics

The question of statistics is the probability theory's inverse: Given the outcomes, what can be said about the process that generated the data?
<br><br>
Often, a distinction is made between two main branches of statistics: descriptive and inferential. Descriptive statistics is concerned with the obtaining, description, and summarisation of data. It is used to talk about the data that has been collected only. Inferential statistics deals with drawing of conclusions from data. Often, it is used to make predictions or comparisons about a larger group (a population) using data gathered about a smaller part of that population. Thus, inferential statistics is used to generalise beyond the data.
<br><br>
In order to draw conclusions from data, it is often necessary to make assumptions about the probabilities of obtaining the different data values. The totality of these assumptions is referred to as a probability model for the data.
At the basis of statistical inference is the formulation of a probability model to describe the data. (Probability model: The mathematical assumptions relating to the likelihood of different data values). Therefore, an understanding of statistical inference requires some knowledge of the theory of probability.
<br>
Prediction, classification, clustering, and estimation are all special cases of statistical inference. Data analysis, machine learning and data mining are various names given to the practice of statistical inference, depending on the context.

## Basic Concepts in Statistics

### Population

***Population*** is a mass of all units under consideration that share common characteristics.

### Parameter

***Parameter*** is a measure of a characteristic of an entire population based on all the elements within that population. A parameter is a statistical constant that describes a feature of a population. Among the examples of population parameters are ***population mean***, ***population standard deviation***, or ***population proportion***.

## Normal Random Variables

We introduce continuous random variables, which are random variables that can take on any value in an interval. We show how their probabilities are determined from an associated curve known as a probability density function. A special type of continuous random variable, known as a normal random variable, is studied. The standard normal random variable is introduced, and a table is presented that enables us to compute the probabilities of that variable. We show how any normal random variable can be transformed to a standard one, enabling us to determine its probabilities.

### Continuous Random Variables

While the possible values of a discrete random variable can be written as a sequence of isolated values, a continuous random variable is one whose set of possible values is an interval. Continuous random variables are random variables that can take on any value in an interval.

Every continuous random variable X has a curve associated with it. This curve, formally known as a probability density function, can be used to obtain probabilities associated with the random variable.

<img align="center" style="zoom: .2" src="http://work.thaslwanter.at/Stats/html/_images/PDF.png">

Probability Density Function (PDF) of a value $x$. The integral over the PDF between $a$ and $b$ gives the likelihood of finding the value of $x$ in that range. That is, the probability that $x$ assumes a value that lies between a and b is equal to the area under the curve between $a$ and $b$:
<br><br>
$P\{a ≤ x ≤ b\} =$ area under curve between $a$ and $b$

### Probability Density Function  (covered in PROBABILITY THEORY)

### Normal Random Variable

The most important type of random variable is the normal random variable. The PDF of a normal random variable $x$ is determined by two parameters:
* the expected value, $\mu = E[x]$
* the standard deviation of $x$, $\sigma = SD(x)$

<img align="center" style="zoom: .3" src="img/NORMAL_PDF.png">

The larger $σ$ is, the more variability there is in thee curve. Not that the curve flattening as $\sigma$ increases.
<br><br>
Because the probability density function of a normal random variable $x$ is symmetric about its expected value $μ$, it follows that $x$ is equally likely to be on either side of $μ$.

#### Standard Normal Random Variable

A normal random variable $x$ having mean value $\mu=0$ and standard deviation $\sigma=1$ is a ***standard normal random variable*** and it is common practice to use letter $z$ to represent a standard normal random variable.
Its density curve is called the ***standard normal curve***:

<img align="center" style="zoom: .2" src="img/STANDARD_NORMAL_CURVE.png">

Approximate areas under a normal curve:

<img align="center" style="zoom: .4" src="img/APPROX_AREAS_NORMAL_CURVE.png">

#### Probabilities Associated with a Standard Normal Random Variable / Standardisation and Z Scores

How to determine ***probabilities*** concerning an arbitrary normal random variable by relating them to probabilities about the standard normal random variable $z$?

$z$ is a standard normal random variable, i.e, $z$ is a normal random variable with mean 0 and standard deviation 1.
The probability that $z$ is between two numbers $a$ and $b$ is equal to the area under the standard normal curve between $a$ and $b$. Areas under this curve have been computed, and tables have been prepared that enable us to find these probabilities.

***Standardisation*** implies using the mean and the standard deviation to generate a standard score (***z-score***) to help us understand where an individual score falls in relation to the other score in the distribution.
<br>
A z-score is a number that indicates how far above or below the mean a given score is in the distribution in the standard deviation units i.e. how many standard deviations a score is above or below the mean. If the raw score is above the mean then the z-score will be positive and if it falls below the mean then it will be negative.

Z-score is given by:

$$
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
$$

where $x$ is a 'raw' standard normal variable.

However if converting a sample mean $\bar{x}$ into a z-score then the raw score is replaced by the sample mean and the standard deviation in the denominator is replaced by the **standard error**.
<br>
Standard error is given by
$$
S E=\frac{\sigma}{\sqrt{n}}
$$
<br><br>
Thus the formula for calculating z score for a sample mean becomes

$$
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
$$

***Z-Table (Standard Normal Table)***

Z-table specifies the probability that $z$ is less or greater than $x$.


```python
def normalProbabilityDensity(x):
    constant = 1.0 / np.sqrt(2*np.pi)
    return(constant * np.exp((-x**2) / 2.0) )

standard_normal_table = pd.DataFrame(data = [],
                                     index = np.round(np.arange(0, 3.5, .1),2),
                                     columns = np.round(np.arange(0.00, .1, .01), 2))

for index in standard_normal_table.index:
    for column in standard_normal_table.columns:
        z = np.round(index + column, 2)
        value, _ = quad(normalProbabilityDensity, np.NINF, z)
        standard_normal_table.loc[index, column] = value

standard_normal_table.index = standard_normal_table.index.astype(str)
standard_normal_table.columns = [str(column).ljust(4,'0') for column in standard_normal_table.columns]

standard_normal_table
```



***Example:***
<br><br>
Let us determine the probability that $z$ is less than $x=1.22$. We can formulate this as $P\{z<1.22\}$.
<br><br>
First, we find the entry in the table corresponding to $x=1.22$ in the left-hand column labeled 1.2. Next, wee search the top row to find the column labeled 0.02. The value found in the cell corresponding to the row and the column is the desired probability of **0.8888**.

***Example:***
<br><br>
Let us determine the probability that $z$ is greater than $x=2$. We can formulate this as $P\{z>2\}$.
<br><br>
Let us not first that $z$ is either less than or equal to 2 or $z$ is greater than 2, and that the two cumulative probabilities amount to 1:
<br><br>
$$P\{z\leq2\} + P\{z>2\} = 1$$
Hence,
$$P\{z>2\} = 1 - P\{z\leq2\} = 1 - 0.9772 = 0.0228$$

#### Percentiles of Normal Random Variables

For any $\alpha$ beetweeen 0 and 1, we define $z_a$ to be that value for which
<br><br>
$$P\{z>z_\alpha\} = \alpha$$
<br><br>
That is, the probability that a standard normal random variable is greater than $z_\alpha$ is equal to $\alpha$.

<img align="center" style="zoom: .3" src="img/NORMAL_RANDOM_VARIABLE_PERCENTILE.png">
<br>
$$P\{z>z_\alpha\} = \alpha$$

Let us determine the value of $z_{\alpha}$ using the z-score table:
<br><br>
$$P\{z<z_{0.0025}\} = 1 - P\{z>z_{0.0025}\} = 1 - 0.025 = 0.975$$
<br><br>
Next, we search the z-score table for the entry 0.975 and find that
<br><br>
$$z_{0.025} = 1.96$$
<br><br>
That is, 2.5 percent of the time, a standard normal random variable will exceed 1.96.
<br><br>
Conversely, 97.5 percent of the time, a standard normal random variable will be less than 1.96. It would be said then that 1.96 is the 97.5 percentile of the standard normal distribution.
<br><br>
Generally, $z_{\alpha}$ is the $100(1-\alpha)$ ***percentile*** of the standard normal distribution.

Let us suppose that would like to find $z_{0.05}$.
<br><br>
$$P\{z<z_{0.05}\} = 1 - P\{z>z_{0.05}\} = 1 - 0.05 = 0.95$$
<br><br>
However, this exact value is not present in the table. Rather, we find
<br><br>
$$P\{z<1.64\} = 0.9495$$
$$P\{z<1.65\} = 0.9505$$
<br><br>
which means that $z_{0.05}$ lies between 1.64 and 1.65, and therefore,
<br><br>
$$z_{\alpha}=1.645$$

## Measures (statistics) in Descriptive Statistics (Summary Statistics)

Measures (*statistics*) are quantities whose values are determined by the data.

### Measures of Centre

Statistics that measure the center or middle value of a data set are **mean**, **median**, and **mode**.

#### Mean

***Sample Mean***

Mean of a sample of $n$ data points $x_1,x_2,...,x_n$ is equal to the arithmetic average of the data values.

$$
\bar{x}=\frac{\sum_{i=1}^n x_i}{n-1}=\frac{x_1+x_2+\cdots+x_n}{n-1}
$$

The mean of a whole population is usually denoted by $μ$, while the mean of a sample is usually denoted by $\bar{x}$.

***Population Mean***

$$
\mu=\frac{\sum_{i=1}^N x_i}{N}
$$

#### Median

***Sample Median***

Sample median, $m$, is a statistic that indicates the center of a data set but unlike the sample mean, it is not affected by extreme values. It is defined as the middle value of the data arranged in numerical order. If the number of data are even, the sample median is the average of the two middle values.

***Population Median***

#### Mode

***Sample Mode***

Sample mode is the data value with the greatest frequency. A data set can have more than one mode; if no single value occurs most frequently, then all the values that occur at the highest frequency are called modal values.

***Population Mode***

### Measures of Dispersion/Variability

Once we have some idea of the center of a data set, the question naturally arises as to how much variation there is. That is, are most of the values close to the center, or do they vary widely about the center? We will discuss the sample variance and sample standard deviation, which are statistics designed to measure such variation.

#### Range

Range is the distance between the smallest and largest data value.

#### Variance

Standard deviation looks at how spread out a group of numbers is from the mean, by looking at the square root of the variance.
The variance measures the average degree to which each point differs from the mean—the average of all data points.

The formula depends on whether the data is being considered a population of its own, or the data is a sample representing a larger population.
<br>
If the data is being considered a population on its own, we divide by the number of data points, $N$.
<br>
If the data is a sample from a larger population, we divide by one fewer than the number of data points in the sample, $n-1$.

***Sample Variance***

IMPORTANT TO DISTINGUISH: VARIANCE IN PROBABILITY THEORY AND IN STATISTICS. In statistics: The variance of a set of n **equally likely values**

$$
s^2=\frac{1}{n-1} \sum_{i=1}^n\left(x_i-\bar{x}\right)^2
$$

the average of the squares of the deviation between each point and the mean.
<br><br>
The squaring is done for two reasons: (1) Dispersion is non-negative. Non-negative values don't cancel out. (2) Squaring amplifies the effect of large differences.

***Population Variance***

$$
\sigma^2=\frac{1}{N} \sum_{i=1}^N\left(x_i-\mu\right)^2
$$

#### Standard Deviation

#### Population Standard Deviation

$$
\sigma=\sqrt{\sigma^2}
$$

#### Sample Standard Deviation

Standard deviation measures the spread of a data distribution. In difference to the variance which amplifies the outliers in the data, standard deviation measures the typical **distance** between each data point and the mean.

Sample standard deviation is given by the positive square root of the sample variance:

$$
s=\sqrt{s^2}
$$

#### Proportion

#### Population Proportion

A population proportion is a fraction of the population that has a certain characteristic.

#### Sample Proportion

### Standardized Moment:  Skeweness, Kurtosis

Two numerical measures of shape – skewness and excess kurtosis – can be used to evaluate normality.

#### Skeweness

$$
\tilde{\mu}_3=\frac{\mu_3}{\sigma^3}=\frac{E\left[(X-\mu)^3\right]}{\left(E\left[(X-\mu)^2\right]\right)^{3 / 2}}
$$

The third standardized moment is a measure of skewness.

Skewness is a measure of the asymmetry of the probability distribution of a random variable about its mean.
Skewness tells about the amount and direction of skew from horizontal symmetry. Its value can be positive or negative, or even undefined.
<br>
If skewness is zero, the data are perfectly symmetrical; not close to zero implies that the data are not normally distributed.

#### Kurtosis

$$
\tilde{\mu}_4=\frac{\mu_4}{\sigma^4}=\frac{E\left[(X-\mu)^4\right]}{\left(E\left[(X-\mu)^2\right]\right)^{4 / 2}}
$$

The fourth standardized moment refers to the kurtosis.

### Pearson Correlation Coefficient

#### Sample Correlation Coefficient

$$
r=\frac{\operatorname{Cov}(X, Y)}{s_x s_y}
$$

#### Population Correlation Coefficient

$$
\rho=\frac{\operatorname{Cov}(X, Y)}{\sigma_X \sigma_Y}
$$

## Statistical Inference

Statistical inference is the science of drawing conclusions about a population based on information contained in a sample.
<br><br>
There are two common **forms of statistical inference**:
<br>
* Estimation
<br>
* Hypothesis Testing

### Estimation

***Estimator*** is a statistic whose value depends on the particular sample drawn. The value of the estimator, called the ***estimate***, is used to infer the value of a population parameter.

There are two common forms of estimation:
<br>
* ***Point estimation***
<br>
* ***Interval estimation***

#### Standard error of an (unbiased) estimator


**The standard deviation of the estimator** is an indication of how close we can expect the estimator to be to the parameter.*

The standard deviation of the sample mean is equal to the population standard deviation divided by the square root of the sample size:

$$
S E=\frac{\sigma}{\sqrt{n}}
$$

#### Point Estimation

***Point estimate*** is a maximally likely single-value of an unknown population parameter.

| Unknown population parameter| Point Estimator | |
|:---|:---|:---|
| Population mean, $\mu$ | Sample mean   | $\bar{x}=\frac{\sum x_i}{n}$|
| Population standard deviation, $\sigma$ | Sample standard deviation  | $s=\frac{\sum{(x_i-\bar{x})}^2}{n-1}$ |
| Population variance, $\sigma^2$ | Sample variance | $s^2=\frac{\sum {(x_i - \bar{x})}^2}{n-1}$ |
| Population proportion, $p$ | Sample proportion | $\hat{p}=\frac{X}{n}$ |

#### Interval Estimate (Confidence Interval)

It cannot be expected that an estimated parameter equals the resulting parameter exactly. Therefore, one can determine an interval about the point estimator in which one can be confident that the parameter lies.
<br><br>
An interval estimate *(confidence interval) of a population parameter is a range of values predicted to contain the parameter. Confidence is the ascribed probability (1-$\alpha$) that the confidence interval actually does contain the population parameter.*

##### Interval Estimators of the Mean of a Normal Population with known Population Variance

***Example*** &nbsp; The case of the interval estimator of a normal mean when the population standard deviation $\sigma$ is assumed known.
<br>
$X_1,...,X_n$ is a sample of size $n$ from a normal population. Standard deviation $\sigma$ is known.
<br>
Let us seek a 95 percent confidence interval estimator for the population mean $\mu$.
<br><br>
First, the sample mean $\bar{X}$ — the point estimator of $\mu$ — is to be determined.

Since we are dealing with a sample mean $\bar{x}$  and not a population mean $\mu$, we will use the following formula which replaces the standard deviation $\sigma$ with the standard error of the mean ($SE=\frac{\sigma}{\sqrt{n}}$):
$$z=\frac{\bar{X}-\mu}{\sigma / \sqrt{n}} = \sqrt{n} \frac{\bar{X}-\mu}{\sigma}$$

Since we know (from the section 'Percentiles of Normal Random Variables') that $z_{0.025} = 1.96$, it follows that 95 percent of the time, the absolute value of $z$ is less than or equal to 1.96.
<br><br>
$$P\left\{\frac{\sqrt{n}}{\sigma} |\bar{X}-\mu| \leq 1.96\right\} = 0.95$$

Multiplying both sides of the inequality by $\sigma/\sqrt{n}$, we obtain
<br><br>
$$P \left\{ |\bar{X}-\mu| \leq 1.96 \frac{\sigma}{\sqrt{n}} \right\} = 0.95$$

From the preceeding, we can say that, with 95 percent probability, $\mu$ and $\bar{x}$ are within 1.96 $\sigma / \sqrt{n}$ of each other. This is equivalent to stating that

$$P \left\{ \bar{X}-1.96\frac{\sigma}{\sqrt{n}} \leq \mu \leq \bar{X}+1.96\frac{\sigma}{\sqrt{n}} \right\} = 0.95$$

or, with 95 percent probability, the interval $\bar{X} \pm 1.96 \ \sigma / \sqrt{n}$ will contain the population mean.

<img align="center" style="zoom: .3" src="img/INTERVAL_ESTIMATION_DEMO_1.png">

$$P\{|z|\leq 1.96\} = P\{-1.96 \leq z \leq 1.96\} = 0.95$$

The interval from $\bar{X} - 1.96 \ \sigma / \sqrt{n}$ to $\bar{X} + 1.96 \ \sigma / \sqrt{n}$ is said to be a 95 percent ***condifence interval estimator*** of the population mean $\mu$. If the observed value of $\bar{X}$ is $\bar{x}$, then we call the interval $\bar{x} \pm 1.96 \ \sigma / \sqrt{n}$ a 95 percent ***confidence interval estimate*** of $\mu$.

### Hypothesis Testing

Statistical hypothesis testing is concerned with using data to test the plausibility of a specified hypothesis.

A **statistical hypothesis** is a statement about the nature of a population. It is often stated in terms of a population parameter.

The null hypothesis, denoted by $H_0$, is a statement about a population parameter. The alternative hypothesis is denoted by $H_1$. The null hypothesis will be rejected if it appears to be inconsistent with the sample data and will not be rejected otherwise.


A **test statistic** (TS) is a statistic whose value is determined from the sample data. Depending on the value of this test statistic, the null hypothesis will be rejected or not.*

In general, if we let TS denote the test statistic, then to complete our specifications of the test, we must designate the set of values of TS for which the null hypothesis will be rejected.

A **critical region** (C), also called a rejection region, is that set of values of the test statistic for which the null hypothesis is rejected.
