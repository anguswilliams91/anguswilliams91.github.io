---
title: "How should we compare models?"
excerpt_separator: "<!--more-->"
categories:
  - Statistics
tags:
  - Statistics
layout: single
classes: wide
---

At various points in the past few years I have had discussions or debates with friends and colleagues about model comparison in the context of Bayesian inference.
What is the most "principled" way to do it?
What are the relative merits of different approaches?
My opinion has evolved alongside my understanding of the subject, and I recently read a paper that conceptually explained some of the intuition that I had developed over the years.
Consequently, I wanted to write a note on this subject in case it is useful for others.

## Introduction to model comparison

Before diving into different approaches for model comparison, let me first define it.
Suppose you are building a statistical model for a particular process, and have some data $$y$$ to use in order to fit and test your approaches.
In the process of analysing the data, you come up with two distinct approaches for modelling: $$M_1$$ and $$M_2$$. 
For example, if you are solving a regression problem, $$M_1$$ might be a GLM of some sort, whereas $$M_2$$ could be a GAM.
Both models seem to fit the data, but you want to quantify which is doing a better job.
Any approach to answering this question falls into the category of *model comparison*.

What should we care about when comparing models?
Fundamentally, we are interested in knowing which model provides a better approximation to the underlying data generating process.
In that sense, we want to know which model *generalises* better beyond the immediate data $$y$$ that we have available.
In other words, if were were to receive some new data $$y_\mathrm{new}$$, which model would describe it better?

The field of model comparison seeks to answer this question, particularly when we must try to make our best guess without access to an unlimited supply of data.

## Recap of the Bayesian approach

No piece of writing about Bayesian methods would be complete without stating Bayes' theorem.
In the context of a single model, we can write this as:

$$p(\theta | y) = \dfrac{p(y | \theta)p(\theta)}{p(y)}, \label{bayes}\tag{1}$$

where $$\theta$$ are the set of model parameters (e.g., the coefficients in a linear regression) and $$y$$ are the data we are analysing.
$$p(\theta | y)$$ is called the *posterior distribution*, because it is the distribution of the model parameters *conditional* on the data $$y$$ (i.e., *after* we receive the data). 
$$p(\theta)$$ is the *prior* because it is the *unconditional* distribution of the model parameters (i.e., prior to seeing the data $$y$$).
$$p(y | \theta)$$ is called the *likelihood*, and is the probability of the data given a particular set of model parameters.
$$p(y)$$ has a few names: the *marginal likelihood* or the *evidence* are probably the two most common.
I'll discuss this part in more detail shortly.

The above discussion is in the context of a single model, so everything is implicitly conditional on the choice of model $$M$$:

$$p(\theta | y, M) = \dfrac{p(y | \theta, M)p(\theta | M)}{p(y | M)}. \label{model_bayes}\tag{2}$$

We don't normally write it like this because it is clear that we are implicitly conditioning on the particular model we are considering, but it's helpful to write Bayes' theorem like this when thinking about model comparison.


## Bayes factors

Since we can use conditional probability to quantify how we should update our beliefs about the parameters $$\theta$$ of an individual model given some data $$y$$, why not use it to update our beliefs about which model might be the best choice from an available set?
This feels like a very natural way to approach the problem of model comparison.
Concretely, we can write down Bayes' theorem as 

$$p(M | y) = \dfrac{p(y | M) p(M)}{p(y)}.$$

Now we have the probability that model $$M$$ is the "true" model given the data at hand: $$p(M | y)$$.
$$p(M)$$ is the prior probability that model $$M$$ is correct.

On the face of it, this seems to be exactly what we need!
Suppose we are comparing two models,$$M_1$$ and $$M_2$$
We can make a very intuitive rule for choosing between them: if $$p(M_1 | y) > p(M_2 | y)$$, then choose $$M_1$$, and choose $$M_2$$ if the converse is true.

If we have no prior preference for $$M_1$$ or $$M_2$$, so that $$p(M_1) = p(M_2) = \frac{1}{2}$$, it's easy to show that

$$\dfrac{p(M_1 | y)}{p(M_2 | y)} = \dfrac{p(y | M_1)}{p(y | M_2)} = K.$$

We call $$K$$ the *Bayes factor*.
In terms of $$K$$, we should choose $$M_1$$ if $$K > 1$$, and choose $$M_2$$ if $$K < 1$$.

So, if we can find a way to calculate $$p(y | M_i)$$, then we can use the Bayes factor to choose between models.
How can we calculate it?
You might have noticed that $$p(y | M)$$ appeared earlier in (\ref{model_bayes}) - it appears as the denominator on the RHS - we called it the *marginal likelihood* or the *evidence*.
But what is it?
A clue comes from one of its names: the *marginal* likelihood.
We can write it as follows:

$$p(y | M) = \int \mathrm{d}\theta\, p(y | \theta, M)\,p(\theta | M) \label{marg_lik}\tag{3}$$

i.e., we *marginalise* out the parameters of the model $$\theta$$ to obtain the probability of the data $$y$$ given the model choice $$M$$. 
We marginalise out $$\theta$$ using the prior distribution, $$p(\theta | M)$$.
This adds some intuition about the Bayes' factor: we choose the model for which the data $$y$$ are *most probable, given all likely configurations of the model parameters*.

This argument for using Bayes' factors for model comparison is quite persuasive, and at first it can seem almost irrefutable because it is so intuitive.

## All that glitters is not gold: Bayes factors are overly sensitive to the prior

Despite seeming apparently watertight at first, the Bayes factor has some undesirable traits.
These are well documented, and I'll focus on just one of them in this note, that they are overly sensitive to the prior.

Practitioners of Bayesian inference quickly learn the rule of thumb that the more data you have, the less influence the prior has on your final inference.
This makes sense - the more evidence you accumulate by collecting more data, the less weight you will place on your prior beliefs. 

Suppose we have a model $$M$$ with a single parameter $$\theta$$ and some data $$y$$.
Let's further assume that the *likelihood* $$p(y | \theta, M)$$ is a normal distribution (when regarded as a function of $$\theta$$) with mean $$\mu_\ell$$ and variance $$\sigma_\ell ^2$$:

$$p(y | \theta, M) = \dfrac{1}{\sqrt{2\pi \sigma_\ell ^ 2}}  \exp [ -\dfrac{(\theta - \mu_\ell)^2}{2 \sigma_\ell ^2}].$$

Further suppose that the prior on $$\theta$$ is a normal distribution with mean zero and variance $$\sigma_\mathrm{prior} ^2$$

$$p(\theta | M) = \dfrac{1}{\sqrt{2\pi \sigma_\mathrm{prior} ^ 2}}  \exp [ -\dfrac{\theta ^2}{2 \sigma_\mathrm{prior} ^2}].$$

Given these assumptions, we can exactly work out some of the quantities of interest for model comparison.
Let's first work out the numerator of the RHS of (\ref{model_bayes}). It turns out to be another normal distribution:

$$p(y | \theta, M)\,p(\theta) = A \times \dfrac{1}{\sqrt{2\pi \sigma_s ^ 2}}\exp [ -\dfrac{(\theta - \mu_\mathrm{post})^2}{2 \sigma_\mathrm{post} ^2}],\label{integral}\tag{4}$$

where

$$\mu_\mathrm{post} = \mu_\ell \dfrac{\sigma_\mathrm{prior} ^2}{\sigma_\mathrm{prior} ^2 + \sigma_\ell ^2} \quad;\quad \sigma_\mathrm{post}^2 = \dfrac{\sigma_\mathrm{prior}^2 \sigma_\ell^2}{\sigma_\mathrm{prior}^2 + \sigma_\ell^2}.\label{norm_post}\tag{5}$$

The constant $$A$$ is equal to yet another normal distribution:

$$A = \dfrac{1}{\sqrt{2\pi (\sigma_\ell ^ 2 + \sigma_\mathrm{prior} ^ 2)}}  \exp [ -\dfrac{\mu_\ell^2}{2 (\sigma_\ell ^2 + \sigma_\mathrm{prior}^2)}].\label{normal_marginal}\tag{6}$$

Now, looking at Bayes' theorem (\ref{model_bayes}), we can see that the left hand side is a probability distribution, which means that it integrates to one:

$$\int p(\theta | y, M) \, \mathrm{d}\theta = 1.$$

Using right hand side of Bayes theorem, we can rearrange this to:

$$\int p(y | \theta, M) p(\theta | M) \, \mathrm{d}\theta = p(y | M).$$

Thus we can see that the marginal likelihood can be regarded as a *normalisation constant* - it guarantees that the posterior distribution integrates to one.
Looking at (\ref{integral}), we can deduce that the constant $$A$$ is in fact the marginal likelihood:

\begin{equation}
\begin{aligned}
\int p(y | \theta, M)\,p(\theta)\,\mathrm{d} \theta &= A \times \int \dfrac{1}{\sqrt{2\pi \sigma_\mathrm{post} ^ 2}}\exp [ -\dfrac{(\theta - \mu_\mathrm{post})^2}{2 \sigma_\mathrm{post} ^2}] \,\mathrm{d}\theta \newline
&= A \newline
\implies p(y | M) &= A 
\end{aligned}
\end{equation}

This result follows from the fact that the normal distribution integrates to one.
This means that the posterior distribution is

$$p(\theta | M, y) = \dfrac{1}{\sqrt{2\pi \sigma_\mathrm{post} ^ 2}}\exp [ -\dfrac{(\theta - \mu_\mathrm{post})^2}{2 \sigma_\mathrm{post} ^2}]$$

So, that seems like a lot of work, but now we have expressions for the posterior $$p(\theta | M, y)$$ and the marginal likelihood $$p(y | M)$$.
Now that we have them, lets's think about the limit when we have a lot of data. 
In this case, the likelihood will be more informative than the prior - it will be more strongly peaked around its mean.
We can express this quantitatively by saying that $$\sigma_\mathrm{prior}^2 \gg \sigma_\ell^2$$, since the larger the variance of a normal distribution, the less strongly peaked it is.

What does the posterior look like in this limit?
We know it is a normal distribution with mean and variance given by (\ref{norm_post}).
In the limit we're interested in, these become:

$$\mu_\mathrm{post} \approx \mu_\ell \quad;\quad \sigma_\mathrm{post}^2 \approx \sigma_\ell^2.$$

This makes intuitive sense - when the likelihood dominates the prior, the posterior will approach the likelihood as the prior loses influence.
What about the marginal likelihood?
Looking at (\ref{normal_marginal}), we can see that this becomes:

$$p(y | M) \approx \dfrac{1}{\sqrt{2\pi\sigma_\mathrm{prior} ^ 2}}  \exp [ -\dfrac{\mu_\ell^2}{2 \sigma_\mathrm{prior}^2}].\label{approx_marginal}\tag{7}$$

One thing to immediately notice is that, in the limit where the likelihood dominates the prior, the posterior distribution is approximately *independent* of the prior: we can see that because it is approximately equal to the likelihood.
But quite the opposite is true of the marginal likelihood!
It explicitly depends on the prior variance $$\sigma_\mathrm{prior}^2$$.

Let's consider two versions of $$M$$.
Both use the same likelihood and data, so they have the same likelihood function, but have different priors.
For concreteness, let's say that $$\sigma_\ell = 1$$, and $$\mu_\ell = 1$$.
In one case, call it $$M_1$$, $$\sigma_\mathrm{prior} = 10$$, and in the other, $$M_2$$, we have $$\sigma_\mathrm{prior} = 100$$.
In both cases, $$\sigma_\ell \ll \sigma_\mathrm{prior}$$, so we're in the limit considered above.
But let's compute the Bayes factor for these two models using (\ref{approx_marginal}):

$$K = \dfrac{p(y | M_1)}{p(y | M_2)} \approx 10.$$

So, even though these two models produce almost *identical* inference (they have almost the same posterior), the Bayes factor tells us that one of them is *ten times* more likely than the other!
Thus, our model selection policy tells us that we should *strongly* favour $$M_1$$ over $$M_2$$, even though these two models produce essentially identical predictions for new data.
At this point, a quote from [Gelman & Shalizi (2012)](http://www.stat.columbia.edu/~gelman/research/published/philosophy.pdf) is appropriate:

"The main point where we disagree with many Bayesians is that we do not see Bayesian methods as generally useful for giving the posterior probability that a model is true, or the probability for preferring model A over model B, or whatever. Beyond the philosophical difficulties, there are technical problems with methods that purport to determine the posterior probability of models, most notably that in models with continuous parameters, aspects of the model that have essentially no effect on posterior inferences within a model can have huge effects on the comparison of posterior probability among models."

## An alternative approach: cross validation

To me, cross validation often used to feel like a concept more commonly used in the field of machine learning, and less elegant than ideas like Bayes factors.
That said, cross validation makes great sense intuitively as a method for model comparison.
Following [Vehtari et al. (2016)](https://arxiv.org/abs/1507.04544), let's first define a useful metric for comparing probabilistic models, the expected log pointwise predictive density for a new dataset:

$$\mathrm{elpd} = \sum_{i=1}^{N} \int \mathrm{d}\tilde y \, p_t(\tilde y_i)\, \log p(\tilde y_i | y, M).\label{elpd}\tag{8}$$

In this expression, we have
 
* $$\tilde y$$ is a new, unseen dataset that (i.e., is not part of the training data $$y$$),assumed to contain $$N$$ data points. 
* $$p_t(\tilde y)$$ is the true data generating distribution of $$\tilde y$$.    
* $$p(\tilde y \| y, M)$$ is the predictive distribution of the model being assessed.  

For Bayesian models, the predictive distribution is

$$p(\tilde y | y, M) = \int \mathrm{d}\theta \, p(\tilde y | \theta, M) p(\theta | y, M).$$

It is so called because this is the pdf we would use for predicting new data, because we include the new information we got from the training data $$y$$.
Notice how similar this looks to the marginal likelihood (\ref{marg_lik}). 
Both are integrals of the product of the likelihood with a distribution over $$\theta$$.
For the marginal likelihood, this distribution is the prior $$p(\theta | M)$$, whereas for the predictive distribution it is the posterior $$p(\theta | y, M)$$.

The elpd looks like a very useful alternative metric for comparing models.
The larger the elpd for a given model, the better we expect it to generalise to new datasets.
Concretely, for two models $$M_1$$ and $$M_2$$, we would choose $$M_1$$ if $$\mathrm{elpd}_\mathrm{M_1} > \mathrm{elpd}_\mathrm{M_2}$$, and choose $$M_2$$ if the converse is true.
Take the example above where the Bayes factor strongly favoured $$M_1$$, but where $$M_1$$ and $$M_2$$ have the same likelihood function and posterior distribution.
What would the elpd look like for these models?
Since they have the same likelihood and posterior distributions, we can say:  

$$
\begin{aligned}
p(\tilde y | y, M_1) &\simeq p(\tilde y | y, M_2) \\
\implies \mathrm{elpd}_{M_1} &\simeq \mathrm{elpd}_{M_2}.
\end{aligned}
$$

Thus, the elpd would say that these two models are the same in terms of predictive performance, and we could choose either one of them.
This makes a lot more sense that the result using Bayes factors!

However, there is a problem - in order to evaluate the elpd, we require the *true data generating distribution*, $$p_t(\tilde y_i)$$.
Obviously we don't know what that is, otherwise we would not be bothering to build a model to approximate it!
The way we proceed is through the use of cross validation.
We can't get at $$p_t(\tilde y_i)$$ directly, but we can note that the training data $$y$$ should be a representative draw from this distribution.
Consequently, we can proceed using Monte Carlo integration to approximate the elpd:

$$\mathrm{elpd} \approx \sum_{i=1}^N \log p(y_i | y_{-i}, M),$$

where $$y_{-i}$$ means "the training set with data point $$i$$ removed".
Using this approximation, we now have a recipe for performing leave-one-out cross validation (LOO CV) for Bayesian models.
For all data points $$y_i$$ in the training set $$y$$, do:

1. Compute the posterior distribution of the model $$M$$ using the dataset with $$y_i$$ removed ($$y_{-i}$$).
2. Calculate the log predictive density of $$y_i$$ using this posterior.
3. Add the result to the approximate elpd being calculated.

This procedure will result in an approximate elpd that can be used to compare two models.
This approach does not have the same problematic sensitivity to the prior as the Bayes factor approach, and is favoured by many (indeed, it's implemented in [R](https://cran.r-project.org/web/packages/loo/index.html) and [python](https://arviz-devs.github.io/arviz/)).
Note that step (1) in the above recipe could be very expensive computationally (imagine carrying out MCMC once per data point for a dataset with 1000 observations.)
Fortunately, [Vehtari et al. (2016)](https://arxiv.org/abs/1507.04544) provide an efficient approximation to this procedure that can alleviate the problem by letting us carry out MCMC just once (or at most a few times).


## Bridging the gap: the marginal likelihood and cross validation

I was recently directed to [Fong & Holmes (2020)](https://academic.oup.com/biomet/article/107/2/489/5715611) which bridges the gap between these two concepts, and provides intuition as to why the marginal likelihood is overly sensitive to the prior.
The paper proves that the marginal likelihood is equivalent to *"leave-$$p$$-out cross validation averaged over all values of $$p$$ and all held-out test sets"*. 
Concretely, this means:

$$\log p(y | M) = \sum_{p=1}^N S_\mathrm{CV}(y; p) \label{result}\tag{9}$$

The cross validation score is 

$$S_\mathrm{CV}(y; p) = \frac{1}{N \choose p}\times \frac{1}{p} \sum_{t=1}^{N \choose p} \log p(y_t | y_{-t}, M).$$

We can pick this apart a bit:

* There are $$N \choose p$$ possible holdout sets of size $$p$$ when there are $$N$$ datapoints, so we average over all of these possible holdout sets.
* For each holdout set, indexed by $$t$$, we evaluate the log predictive density of the holdout set $$y_t$$, conditioned on the full dataset with this holdout set removed, $$y_{-t}$$.

This looks very similar to (\ref{elpd}), as you'd expect.
The result (\ref{result}) draws a concrete connection between cross validation and the marginal likelihood.
In the previous section, I only considered *leave-one-out* cross validation.
But the marginal likelihood considers *all possible holdout set sizes*.
This explains why the two quantities behave differently (in the toy example I gave, the marginal likelihood favoured $$M_1$$ strongly, whereas LOO CV could not choose between $$M_1$$ and $$M_2$$).
In particular, it explains why the marginal likelihood can be sensitive to the prior.
Fong & Holmes put it well (I have modified the equation numbers and the notation slightly to match this post):

"The last term on the right-hand side of (\ref{result}),    
$$S_\mathrm{CV}(y;N) = \sum\limits_{i=1}^N \log \int \,p(y_i | \theta, M)\,p(\theta)\,\mathrm{d}\theta$$⁠,  
 involves no training data and scores the model entirely on how well the analyst is able to specify the prior. In many situations, the analyst may not want this term to contribute to model evaluation. Moreover, there is conflict between the desire to specify vague priors to safeguard their influence and the fact that diffuse priors can lead to an arbitrarily large and negative model score for real-valued parameters from (\ref{result}). It may seem inappropriate to penalize a model based on the subjective ability to specify the prior, or to compare models using a score that includes contributions from predictions made using only a handful of training points even with informative priors."

Fong & Holmes go on to recommend an approach where, instead of considering all sizes of holdout set between one and $$N$$, analysts can consider a maximum size holdout set that is $$< N$$ in order to avoid the problem above. 
LOO CV is such an approach, where the maximum size holdout set is one.

## Conclusion

I really enjoyed seeing the result from [Fong & Holmes (2020)](https://academic.oup.com/biomet/article/107/2/489/5715611), because it joined the dots between different concepts in model comparison that I have learned about.
It reinforced the point that the marginal likelihood may not be appropriate for comparing models, and gave an elegant explanation as to why that is.
For me, a simple conclusion is: when comparing models, try to replicate as closely as possible the way that the model will be used in the future.
I'll continue to use various flavours of cross validation to compare my models!