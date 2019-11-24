---
title: "Enriched prior structure for Hierarchical models"
excerpt_separator: "<!--more-->"
categories:
  - Statistics
  - Sport
tags:
  - Statistics
  - Sport
  - Stan
layout: single
classes: wide
---
One of the great things about Bayesian modelling and probabilistic programming is the ease with which we can build models with rich prior structures.
Often, we see big improvements in modelling results by introducing relatively simple hierarchical structure in the prior: but we can go further than that if the our domain knowledge or the data suggest we should.
This post is a small case study of where I start with simple hierarchical priors and iteratively develop richer prior structure by model checking and expansion.

## Context

I recently attended [StanCon](https://mc-stan.org/events/stancon2019Cambridge/), which was great fun.
Besides the talks and tutorials, there was also a competition: given some historical data from football leagues, the aim was to build a model that could predict the scorelines of a holdout set of data.

Quantitatively, the task is to build a generative model for the final scoreline of football matches.
This means assigning a probability to the tuple $$(G_h, G_a)$$, where $$G_h$$ is the number of goals scored by the home team, and $$G_a$$ is the number of goals scored by the away team.

I decided to take part, and in the process made some small changes to a well-known model for this kind of data.
For the purposes of this post, I grabbed last season's Premier League results to use as data.

## The likelihood

Before discussing the prior, first I'll describe the likelihood function that I'll use for the model. 
For this, I will use a classic reference for statistical models of football matches: [Dixon & Coles (1997)](http://web.math.ku.dk/~rolf/teaching/thesis/DixonColes.pdf).

Since the number of goals scored by either team are small, positive integers, the Poisson distribution is a logical first choice of likelihood function.
The likelihood of a scoreline in their model is:

$$
\mathrm{Pr}\left(G_h,\,G_a | \lambda,\mu\right) 
= \tau(G_h,G_a; \lambda, \mu)
\times\mathrm{Poisson}(G_h;\,\lambda)
\times\mathrm{Poisson}(G_a;\,\mu),
\tag{1}
\label{likelihood}
$$

where the Poisson distrbutions have rates $$\lambda$$ and $$\mu$$.
There is also a further term, $$\tau(G_h,G_a; \lambda, \mu)$$, that breaks the conditional independence between $$G_h$$ and $$G_a$$.
Dixon & Coles introduced this term because it improved model performance for low-scoring matches.
For the sake of simplicity, I'll remove this term for the model (it adds a little more complexity to the Stan model, but does not relate to the goal of this post).

The main question is: what values should we give the rates, $$\lambda$$ and $$\mu$$? 
These characterise the expected number of goals each team will score.
$$\lambda$$ is the expected number of goals that the home team will score, and likewise for $$\mu$$ the expected number for the away team.

Intuitively, these quantities should depend on both of the teams.
Man City score a lot of goals, but we would expect them to score fewer against  Liverpool than they would against Aston Villa, for example.
To encode this intuition, the rates $$\lambda$$ and $$\mu$$ are written as follows:

$$
\log \lambda = \alpha_h - \beta_a + \gamma, \\
\log \mu = \alpha_a - \beta_h.
\tag{2}
\label{rates}
$$

For every team in the league, we have introduced an *attacking ability* $$\alpha$$ and a *defending ability* $$\beta$$.
In addition, the home team rate is boosted by a factor of $$\gamma$$, which is a *home advantage* parameter.
Clearly this structure encodes the intuition above.
If Man City have a large $$\alpha$$, but the number of goals they'll score is modulated by the $$\beta$$ of their opponent (larger for Liverpool, smaller for Aston Villa).

## Priors

So, in Equation \eqref{likelihood} we have our likelihood.
It is parameterised by $$\theta = (\{\alpha_i\}, \{\beta_i\}, \gamma)$$, where $$\{\alpha_i\}$$ means the set of attacking abilities for each of the teams in the league, and likewise for the defensive abilities.
In the original paper, Dixon & Coles used maximum likelihood to infer point estimates for the parameters, so their version of the model is now fully specified.

To do Bayesian inference, we of course use Bayes' theorem to obtain a posterior on the model parameters $$\theta$$:

$$p(\theta \,|\, \{G_{h,i}, G_{a,i}\}) 
\propto p(\theta)\prod_{i=1}^{N_\mathrm{match}} Pr(G_{h,i}, G_{a,i} \,|\, \theta).$$

In the above, I've introduced an additional index $$i$$ to label the particular match in question.
The product on the right implies that I have assumed that the matches are conditionally independent given $$\theta$$.

Now, the job is to specify the prior, $$p(\theta)$$.
For the rest of the post, I'll discuss the development of this object.
Throughout, I use a simple prior for $$\gamma$$

$$\gamma \sim \mathrm{Lognormal}(0, 1),$$

and only discuss priors for the attack and defence abilities.

### 1. Hierarchical, independent prior for $$\alpha$$ and $$\beta$$

A good first step is to make the model *hierarchical* by presuming that the attack and defence abilities are drawn from a common prior distribution.
This is written as

$$ 
\alpha_j \sim \mathcal{N}\left(\mu_\alpha, \sigma_\alpha\right), \\
\beta_j \sim \mathcal{N}\left(\mu_\beta, \sigma_\beta\right).
$$

where we've introduced a further set of *hyper parameters* $$\phi = (\mu_\alpha, \mu_\beta, \sigma_\alpha, \sigma_\beta)$$.
It's then helpful to re-write Bayes theorem:

$$
p(\theta, \phi \,|\, \{G_{h,i}, G_{a,i}\}) 
\propto 
p(\theta \,|\, \phi)\,p(\phi)\,
\prod_{i=1}^{N_\mathrm{match}} Pr(G_{h,i}, G_{a,i} \,|\, \theta).
$$

This makes it clear that we also need a prior $$p(\phi)$$ on our hyper parameters.
I'll specify these as follows:

$$
\mu_{\alpha} = 0, \\
\mu_{\beta} \sim \mathcal{N}(0, 1), \\
\sigma_{\alpha, \beta} \sim \mathcal{N}^+(0, 1).
$$

Looking at Equation \eqref{rates}, we can see that $$\mu_{\alpha}$$ can be set to zero, because an $$\alpha$$ only ever appears in the likelihood when summed with a $$\beta$$.
I then use a unit normal prior for $$\mu_{\beta}$$ and half-normal priors for the variances.
Our first attempt at specifying a model is done!
Now to fit it in Stan.
Here's the code for this model:

```text
data {
    int<lower=1> nteam;
    int<lower=1> nmatch;
    int home_team[nmatch];
    int away_team[nmatch];
    int home_goals[nmatch];
    int away_goals[nmatch];
}
parameters {
    vector[nteam] a;
    vector[nteam] b;
    real<lower=0> gamma;
    real<lower=0> sigma_a;
    real<lower=0> sigma_b;
    real mu_b;
}
model {
    // rates for the two poisson distributions
    vector[nmatch] lambda = a[home_team] + b[away_team] + gamma;
    vector[nmatch] mu = a[away_team] + b[home_team];

    //hyper priors
    sigma_a ~ normal(0, 1);
    sigma_b ~ normal(0, 1);
    mu_b ~ normal(0, 1);

    // priors
    gamma ~ lognormal(0, 1);
    a ~ normal(0, sigma_a);
    b ~ normal(mu_b, sigma_b);

    // likelihood
    home_goals ~ poisson_log(lambda);
    away_goals ~ poisson_log(mu);
}
```

Stan's expressive syntax means that this code looks quite similar to the formulae I used to describe the model!
In practice, I would have used a non-centered parameterisation of this model in order to help Stan's sampler, but I decided not to here for clarity.








