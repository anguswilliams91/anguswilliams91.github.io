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

The mens rugby union world cup is just starting[^1], so I thought it would be fun to make some predictions before the tournament gets going.
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

I will model the number of points scored by each of the teams in a match using an independent negative binomial model[^2]:

$$
\mathrm{points_{ij}} \sim \mathrm{NegBinom2}(\alpha_i \beta_j, \phi).
$$

$$\mathrm{points}_{ij}$$ refers to the number of points scored by team $$i$$ against team $$j$$.
Each team is assigned an *attacking aptitude* $$\alpha_i$$ and a *defending aptitude* $$\beta_i$$.
The expected number of points scored by team $$i$$ against team $$j$$ is then equal to the product of team $$i$$'s attacking aptitude with team $$j$$'s defending aptitude.
This is fairly intuitive: the better team $$i$$ is at attacking (larger $$\alpha_i$$), the more points they'll score.
The better team $$j$$'s defending aptitude (smaller $$\beta_j$$), the fewer points team $$i$$ will score.

This kind of model is very commonly used in the context of football matches (check out the classic [Dixon & Coles](http://web.math.ku.dk/~rolf/teaching/thesis/DixonColes.pdf) paper).[^3]
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
Here's a figure like that, where I plot the distribution of total points scored in a match:

![points-ppc]({{site.github.url}}/assets/images/rugby_wc_post/total_points_distro.png){:class="img-responsive"}

and another where I plot the distribution of the difference in points between the two teams:

![diff-ppc]({{site.github.url}}/assets/images/rugby_wc_post/difference_distro.png){:class="img-responsive"}

The model seems to consistently produce a few matches with *very* high points totals relative to the data, but otherwise seems to be a reasonable representation of the data.

## Simulating the world cup

Now that I have posterior samples from the model, I can simulate the world cup many times and use the results to evaluate the probability that each of the teams will win.
To do this, I need to know the rules of the world cup.
In the group stages, teams are allocated 4 points for a win, 2 points for a draw and 0 points for a loss.
Additionally, teams are awarded a bonus point if they score 4 or more tries, or if they lose by 7 or fewer points.
Since my model produces total points, but does not predict the number of tries a team will score explicitly, I just allocate a bonus point if they score more than 25 points.

My recipe will be as follows:
1. Select a set of model parameters $$\theta = (\{\alpha_i\}, \{\beta_i\}, \phi)$$ from a single iteration of MCMC.
2. Use the parameters to simulate a single realisation of each of the group matches, and use the rules of the tournament to figure out which teams will graduate into the knockout stages.
3. Simulate each of the knockout stage matches, eventually ending up with a winner.
4. Store the results, and repeat (1) to (3) for every iteration of MCMC.

Once these calculations have been done, I have thousands of simulated world cups.
To calculate the posterior predictive probability of a given team winning, all I have to do is calculate the fraction of times that team won in my simulations -- simple!
I really like this side of using MCMC, it becomes straightforward to calculate approximate posterior predictive distribution of non-trivial functions.

## Results

Ok -- so who is going to win?
Here are the probabilities assigned to each of the teams by the model (I only display probabilities for teams for whom the probability is 0.01 or larger):

|--------------------------|--------------------------------------|
| Team                     | Probability of winning the world cup |
|--------------------------|--------------------------------------|
| New Zealand              | 0.50                                 |
| England                  | 0.15                                 |
| South Africa             | 0.13                                 |
| Ireland                  | 0.08                                 |
| Wales                    | 0.06                                 |
| Australia                | 0.05                                 |
| France                   | 0.01                                 |
| Scotland                 | 0.01                                 |
|--------------------------|--------------------------------------|


New Zealand are massive favourites, with England, South Africa and Ireland all hovering at around 0.1 chance of winning.
I am perhaps surprised by Wales being a assigned a noticeably lower probability than England, but perhaps this is due to the likely path they would need to take to the final being more difficult than England's.
Also, Wales had a dire spell a few years ago, and the simple model I used does not account for changing form, so it might underrate Wales somewhat.

As an England fan, I am also curious to know how likely England are to get to various stages of the tournament.
The model gives England a probability of 0.94 of getting out of the group -- so there's a very good chance they'll do better than at the last world cup, and we should get to see them in a quarter final.
The probability of them getting to the semi finals is 0.65, and 0.30 for the final.
So, fans should be disappointed if they don't see England win at least one knockout match!

Whilst I'm at it, here are the model outputs for each of the groups (again I leave out teams with probability < 0.01, and round up to the nearest 0.01).

### Group A

| Team     | Probability of winning group A |
|----------|--------------------------------|
| Ireland  | 0.71                           |
| Scotland | 0.25                           |
| Japan    | 0.03                           |
| Samoa    | 0.01                           |

| Team     | Probability of being runner up of group A |
|----------|-------------------------------------|
| Scotland | 0.50                                |
| Ireland  | 0.24                                |
| Japan    | 0.18                                |
| Samoa    | 0.08                                |

### Group B

| Team         | Probability of winning group B |
|--------------|--------------------------------|
| New Zealand  | 0.75                           |
| South Africa | 0.25                           |

| Team         | Probability of being runner up of group B |
|--------------|-------------------------------------|
| South Africa | 0.72                                |
| New Zealand  | 0.25                                |
| Italy        | 0.03                                |

### Group C

| Team                     | Probability of winning group C |
|--------------------------|--------------------------------|
| England                  | 0.74                           |
| France                   | 0.14                           |
| Argentina                | 0.11                           |

| Team                     | Probability of being runner up of group C |
|--------------------------|-------------------------------------|
| France                   | 0.41                                |
| Argentina                | 0.35                                |
| England                  | 0.20                                |
| United States of America | 0.03                                |
| Tonga                    | 0.02                                |

### Group D

| Team      | Probability of winning group D |
|-----------|--------------------------------|
| Wales     | 0.49                           |
| Australia | 0.45                           |
| Fiji      | 0.06                           |

| Team      | Probability of being runner up of group D |
|-----------|-------------------------------------|
| Australia | 0.41                                |
| Wales     | 0.38                                |
| Fiji      | 0.15                                |
| Georgia   | 0.05                                |

Three of the four groups have a clear favourite. 
Group D, unsurprisingly, is a toss-up between Wales and Australia to top the group.

## Conclusions

This was as nice end-to-end bit of analysis: scraping the data, building the model and then simulating the results.
The conclusions are slightly at odds with the articles I've seen about the competition, which claim that it is very open and a few teams have a relatively even chance of winning.
According to this model, New Zealand are still huge favourites.

Nonetheless, England's chances aren't too bad, so there's still hope!

[^3]: I actually have a python package on [GitHub](https://github.com/anguswilliams91/bpl) that implements this model for football.
[^2]: I am using the alternative parameterisation of the negative binomial, in terms of its expectation $$\mu$$ and dispersion parameter $$\phi$$.
[^1]: I am a bit late in posting this, and a few group stage matches have already happened!