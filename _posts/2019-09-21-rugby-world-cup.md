---
title: "Who will win the rugby world cup?"
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

The mens rugby union world cup is just starting, so I thought it would be fun to make some predictions before the tournament gets going.
The plan is to build a statistical model using previous match results, and then use it to evaluate the probability that each of the teams will win.
(All of the code I used to obtain the data and produce the results in this post are [here](https://github.com/anguswilliams91/ruwc_2019).)

## Historical data

I have modelled football data before, and have found it very easy to obtain.
Rugby data, on the other hand, proved a bit trickier to get hold of.
I couldn't find a website where I could simply download a file containing historical results, so I had to resort to scraping ESPN.
I hadn't done this in a while, so it took a little while to inspect the html and figure out how to extract the data.
In the end, I downloaded all of the mens international rugby union results between the present day and 1st January 2013 to use as a training set for my model.
A combination of the python standard library and Beautiful Soup got me there in the end!

## Model

I will model the number of points scored by each of the teams in a match using an independent negative binomial model:

$$\mathrm{points_{ij}} \sim \mathrm{NegBinom}(\alpha_i \beta_j, \phi).$$

$$\mathrm{points}_{ij}$$ refers to the number of points scored by team $$i$$ against team $$j$$.
Each team is assigned an *attacking aptitude* $$\alpha_i$$ and a *defending aptitude* $$\beta_i$$.
The expected number of points scored by team $$i$$ against team $$j$$ is then equal to the product of team $$i$$'s attacking aptitude with team $$j$$'s defending aptitude.
This is fairly intuitive: the better team $$i$$ is at attacking (larger $$\alpha_i$$), the more points they'll score.
The better team $$j$$'s defending aptitude (smaller $$\beta_j$$), the fewer points team $$i$$ will score.

This kind of model is very commonly used in the context of football matches (check out the classic [Dixon & Coles](http://web.math.ku.dk/~rolf/teaching/thesis/DixonColes.pdf) paper).[^1]
In that case, a Poisson likelihood is typically used, but I found that a negative binomial better replicated the distribution of scores in rugby matches.

I also use a hierarchical prior on the attack and defense aptitudes, e.g.:

$$\log \alpha_i \sim \mathcal{N}(\mu_\alpha, \sigma_\alpha).$$

This should regularise the model better.
Here's the Stan code for the model:

```text
data {
    int<lower=1> nteam;
    int<lower=1> nmatch;
    int home_team[nmatch];
    int away_team[nmatch];
    int home_points[nmatch];
    int away_points[nmatch];
}
parameters {
    vector[nteam] log_a_tilde;
    vector[nteam] log_b_tilde;
    real<lower=0> sigma_a;
    real<lower=0> sigma_b;
    real mu_b;
    real<lower=0> phi;
}
transformed parameters {
    vector[nteam] a = exp(sigma_a * log_a_tilde);
    vector[nteam] b = exp(mu_b + sigma_b * log_b_tilde);
    vector[nmatch] home_rate = a[home_team] .* b[away_team];
    vector[nmatch] away_rate = a[away_team] .* b[home_team];
}
model {
    phi ~ normal(0, 5);
    sigma_a ~ normal(0, 1);
    sigma_b ~ normal(0, 1);
    mu_b ~ normal(0, 5);
    log_a_tilde ~ normal(0, 1);
    log_b_tilde ~ normal(0, 1);
    home_points ~ neg_binomial_2(home_rate, phi);
    away_points ~ neg_binomial_2(away_rate, phi);
}
generated quantities {
    int home_points_rep[nmatch];
    int away_points_rep[nmatch];
    for (i in 1:nmatch) {
        home_points_rep[i] = neg_binomial_2_rng(home_rate[i], phi);
        away_points_rep[i] = neg_binomial_2_rng(away_rate[i], phi);
    }
}
```

If the implementation looks a bit funny, that's because I used a [non-centered version](https://arxiv.org/abs/1312.0906) of the model so that Stan's sampler would work better.


## Model checks

In the interest of brevity, I won't spend long on this.
But, it would be pretty bad practice not to spend *some* time showing that the model produces reasonable simulated data!
In the above Stan code you can see that I generate some simulated data for this purpose.
I end up with the same number of simulated datasets as there are steps in my MCMC chain.

One nice way to do visual checks is to plot the distribution of the data on the same axes as the distribution of a single simulated dataset.
Since we have lots of simulated datasets, we can make this plot multiple times.
This gives us an idea if the real data are "typical" of the model.
Here's a figure like that, where I plot the distribution of points scored by one of the teams in the match:

![points-ppc]({{site.github.url}}/assets/images/rugby_wc_post/points_distro.png){:class="img-responsive"}

and another where I plot the distribution of the difference in points between the two teams:

![diff-ppc]({{site.github.url}}/assets/images/rugby_wc_post/difference_distro.png){:class="img-responsive"}

The simulated datasets look reasonable when compared to the real data.

## Simulating the world cup

Now that I have posterior samples from the model, I can simulate the world cup many times and use the simulations to evaluate the probability that each of the teams will win.




[^1]: I actually have a python package on [GitHub](https://github.com/anguswilliams91/bpl) that implements this model for football.