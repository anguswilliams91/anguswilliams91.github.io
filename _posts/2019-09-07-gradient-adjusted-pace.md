---
title: "Tracking my fitness"
excerpt_separator: "<!--more-->"
categories:
  - Statistics
  - Sport
tags:
  - Statistics
  - Sport
  - running
  - Stan
---

I like to go running, and I also like to see how I'm doing by using the Strava app.
It's satisfying to see progress over time if I'm training for an event.

This is pretty easy to do if I am repeating the same routes.
However, I recently moved house, and consequently have a new set of typical routes.
A big difference in my new surroundings is that it's *really hilly*!
This means that my typical pace now comes out slower than before the move.

Fortunately, Strava has a nifty feature called *Gradient Adjusted Pace* (GAP).
GAP estimates of your pace are corrected for the vertical gradient of the terrain you're running on.
This facilitates comparison between runs that took place in different locations -- great!

A lazy search didn't instantly tell me how GAP is calculated, so I thought it would be fun to try and come up with my own recipe instead.
In the process, I also wondered if I could track some underlying metric for my fitness.

## What is GAP?

I haven't actually got a rigorous definition of GAP yet.
If I want to do something statistical, I had better come up with one.
What about this:

*"If I had done the same run in a parallel universe where all of the hills were removed, but everything else stayed the same, how fast would I have been?"*

This would let me put all of my runs of the same distance on a (literally) level playing field for comparison.
This definition also means that what I'm trying to calculate is a *counterfactual*, so I'm doing causal inference.

## Data

To come up with my own GAP estimates, I'll use data from Strava (there's an API that you can use to download your data).
To keep things simple, I'll model the **average pace** of each run as a function of the **elevation gain** during the run and the **total distance** of the run.
All of my runs are loops, so using elevation gain takes into account that I lose as much elevation as I gain.

Here's a plot of the data:
![image-title-here]({{site.url}}/assets/images/strava_post/distance_vs_speed.png){:class="img-responsive"}

The relationship between the log of distance and the log of speed looks like it would be well modelled as a straight line with some scatter.
As expected, my average pace is lower when I run further.
The points are coloured by the elevation gain, which lets us see, as we'd expect, that I get slower if there's more elevation gain in a run.
I've got data from 128 runs in total.

## Statistical model

Based on the visualisation above, I'll use the following statistical model

$$\log\,(\mathrm{pace}) =

\alpha 
+ \beta_\mathrm{elevation}\log\left(\mathrm{elevation} + \delta\right) 
+ \beta_\mathrm{distance}\log\left(\mathrm{distance}\right)
+ \epsilon
$$

where $$\epsilon$$ is normally distributed noise with some variance $$\sigma ^2$$.
Note that $$\delta$$ is not a parameter: it's a constant that I add to elevation so that I don't run into trouble with logs when elevation is zero.
I wrote this model in Stan:
```stan
data {
  int n;
  vector[n] log_pace;
  vector[n] log_elevation;
  vector[n] log_distance;
}
parameters {
  real beta_elevation;
  real beta_distance;
  real<lower=0> sigma;
}
transformed parameters {
  vector[n] z = beta_elevation * log_elevation 
                + beta_distance * log_distance;
}
model {
  beta_elevation ~ normal(0, 1);
  beta_distance ~ normal(0, 1);
  sigma ~ normal(0, 5);
  log_pace ~ normal(z, sigma);
}
generated quantities {
  vector[n] log_pace_rep;
  for (i in 1:n) {
    log_pace_rep[i] = normal_rng(z[i], sigma);
  }
}
```
I centred log speed, so there's not explicit intercept term.
I also generate some simulated data for model checking.
The parameters of interest are the two betas:
![image-title-here]({{site.url}}/assets/images/strava_post/model_params.png){:class="img-responsive"}

Both are negative, as expected.
Armed with this model, I can now calculate my simple GAP estimates through a simple re-arranging of my linear model and setting elevation to zero:

$$\log (\mathrm{GAP}) = \log(\mathrm{pace}) + \beta_\mathrm{elevation}\left[\log(\epsilon) - \log(\mathrm{elevation} + \delta) \right].$$

Because I have uncertainty about the value of $$\beta_\mathrm{elevation}$$, I'll also have uncertainty about the GAP value that I infer for each run.
The more elevation there is in a run, the more uncertainty there will be in the estimate of GAP.
To compute this uncertainty, I can just plug my MCMC samples for $$\beta_\mathrm{elevation}$$ into the above formula.
Here's the result of estimating GAP for all of my runs:
![image-title-here]({{site.url}}/assets/images/strava_post/gap_vs_true.png){:class="img-responsive"}

In this figure, I'm plotting GAP against my actual pace for the run.
The points are coloured by the elevation gain of the run.
The error-bars are the 95% credible interval for GAP.

The plot broadly makes sense: the GAP estimates are generally lower than the true pace (i.e., I would have run faster on the flat).
Furthermore, runs with more elevation have a bigger difference between GAP and my actual pace.
I couldn't see how to get Strava's GAP out of the API, so I didn't do a full comparison between the average GAP produced by Strava and my simple model.
I grabbed Strava's GAP for the run with the largest elevation gain and did a comparison there:
![image-title-here]({{site.url}}/assets/images/strava_post/gap_vs_strava.png){:class="img-responsive"}

At least in this case, my approach and the Strava data are consistent with one another.

## Tracking fitness

The simple approach to calculating GAP produced somewhat reasonable results.
But, given my aim of summarising my fitness over time, there's a bit more work to do.
The simple model above does not allow for variation in fitness -- it just says that my pace is a function of how far I'll go and how hilly the run is.
To include some notion of fitness, I expanded the model so that it looks like this:

$$\log\,(\mathrm{pace}_i) =

\alpha_i 
+ \beta_\mathrm{elevation}\log\left(\mathrm{elevation} + \delta\right) 
+ \beta_\mathrm{distance}\log\left(\mathrm{distance}\right)
+ \epsilon
$$

where the index $$i$$ encodes an ordering to my runs (e.g. run 2 comes after run 1 and before run 3).
The key difference between this model and the original one is that now the intercept $$\alpha$$ is a function of time instead of a constant.
I should probably index using time explicitly, but I was too lazy for that.
Since I don't expect my fitness to vary wildly between runs, I then put a random walk prior on the intercepts:

$$\alpha_i = \alpha_{i - 1} + \zeta$$,

where $$\zeta$$ is normally distributed noise with variance $$\sigma_\mathrm{rw}^2$$.
Now, I can infer the set $$\{\alpha_i\}$$ and interpret them and my "fitness" on each of my runs.
Here's the stan code for this model:
```stan
data {
  int n;
  vector[n] log_pace;
  vector[n] log_elevation_gain;
  vector[n] log_distance;
}
parameters {
  real beta_elevation;
  real beta_distance;
  real<lower=0> sigma;
  vector[n] fitness_std;
  real<lower=0> sigma_rw;
}
transformed parameters {
  vector[n] fitness;
  vector[n] z;
  fitness[1] = fitness_std[1];
  for (i in 2:n) {
    fitness[i] = fitness_std[i] * sigma_rw 
                 + fitness[i - 1];
  }
  
  z = fitness 
      + beta_elevation * log_elevation_gain 
      + beta_distance * log_distance;
}
model {
  beta_elevation ~ normal(0, 1);
  beta_distance ~ normal(0, 1);
  sigma ~ normal(0, 1);
  log_pace ~ normal(z, sigma);
  fitness_std ~ normal(0, 1);
  sigma_rw ~ normal(0, 1);
}
generated quantities {
  vector[n] log_speed_rep;
  for (i in 1:n) {
    log_speed_rep[i] = normal_rng(z[i], sigma);
  }
}
```
After running MCMC, I can plot my inferred fitness over time:
![image-title-here]({{site.url}}/assets/images/strava_post/fitness_trend.png){:class="img-responsive"}

The grey band is the 68% credible intervals, and the black dots mark when a run took place.
The results look broadly as I would expected them to. 
I know I was pretty fit last summer, but then had an injury which bothered me until early January.
I then started training again for a couple of half-marathons in May / June.
A definite issue with this approach is that the amount of effort I put into runs is variable, but the model assumes that I am trying my best in every run.
One way to get around this would be to include heart-rate data, which provide a measure of how strained I am during the run.
But as a simple first approach, the results are reasonable.

I can also check if the model is reproducing the data reasonably:
![image-title-here]({{site.url}}/assets/images/strava_post/residuals.png){:class="img-responsive"}

Each of the box-and-whisker plots in the figure show the distribution of the difference between the simulated pace from my model posterior samples and the actual pace.
All being well, these residuals should largely line up around zero.
Since there are a bunch of runs, we would expect to see some where the difference between the simulations and the actual pace is larger than this just because of statistical noise.
Broadly, the model seems to fit the data relatively well according to this figure.
There looks like there might be some autocorrelation (i.e. runs where the model underestimates the pace tend to be clustered near to each other, you can see this is a bit in the centre of the plot), but broadly speaking it looks ok.