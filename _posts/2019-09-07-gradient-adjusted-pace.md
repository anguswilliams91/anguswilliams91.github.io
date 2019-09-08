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
layout: single
classes: wide
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

## What is GAP?

To get going, I need a definition for GAP.
Let's go with this:

*"If I had done the same run in a parallel universe where all of the hills were removed, but everything else stayed the same, how fast would I have been?"*

This achieves the desired goal by putting all my runs on a (literally) level playing field for comparison.
This definition also means that what I'm trying to calculate is a *counterfactual*, so I'll need to interpret whatever model I build as causal.
To produce estimates of the answer to this question, I will need to use data from my previous runs to build a statistical model, which I can then query.

## Data

To come up with my own GAP estimates, I'll use data from Strava (there's an API that you can use to download your data).
To keep things simple, I'll model the **average speed** of each run as a function of the **elevation gain** during the run and the **total distance** of the run.
All of my runs are loops, so using elevation gain takes into account that I lose as much elevation as I gain.

Here's a plot of the data:

![distance-vs-speed]({{site.github.url}}/assets/images/strava_post/distance_vs_speed.png){:class="img-responsive"}

The relationship between the log of distance and the log of speed looks like it would be well modelled as a straight line with some scatter.
As expected, my average speed is lower when I run further.
The points are coloured by the elevation gain, which lets us see that I get slower if there's more elevation gain in a run.
I've got data from 128 runs in total.

## Statistical model

Based on the visualisation above, I came up with the following super-simple linear model

$$\log\,(\mathrm{speed}) =

\alpha 
+ \beta_\mathrm{elevation}\log\left(\mathrm{elevation} + \delta\right) 
+ \beta_\mathrm{distance}\log\left(\mathrm{distance}\right)
+ \epsilon
$$

where $$\epsilon$$ is normally distributed noise with variance $$\sigma ^2$$.
Note that $$\delta$$ is not a parameter: it's a constant that I add to elevation so that I don't run into trouble with logs when elevation is zero.
I set $$\delta = 10\mathrm{m}$$.
I don't have much data, so point estimates of the model parameters aren't going to cut it.
To properly quantify my uncertainty, I'll take a Bayesian approach and sample the posterior using [Stan](mc-stan.org).

Here's the stan code for this model:

```stan
data {
  int n;
  vector[n] log_speed;
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
  log_speed ~ normal(z, sigma);
}
generated quantities {
  vector[n] log_speed_rep;
  for (i in 1:n) {
    log_speed_rep[i] = normal_rng(z[i], sigma);
  }
}
```

I centred log speed, so there's no explicit intercept term.
I also generate some simulated data to be used for model checking in the `generated quantities` block.
The marginal posterior distributions for the slopes look like this:

![slopes-posterior]({{site.github.url}}/assets/images/strava_post/model_params.png){:class="img-responsive"}

Both elevation gain and total distance cause my average speed to go down, as expected.
Let's also do a quick check of the model by plotting the distribution of the residual  

$$\log(\mathrm{speed}) - \log(\mathrm{speed})_\mathrm{rep}$$

for each of the runs.
$$\log(\mathrm{speed})_\mathrm{rep}$$ are the simulated speeds from the model posterior predictive distribution.
For each run (in temporal order), I display a box and whisker plot of these residuals.
The speeds and simulated speeds are scaled according to the mean and variance of the observed data, so we should expect to see the residuals distributed something like a unit normal, with some variation between runs.

![posterior-checks]({{site.github.url}}/assets/images/strava_post/residuals_simple.png){:class="img-responsive"}

The figure shows that the model is doing a reasonable job at replicating the data, but there is obviously some autocorrelation visible (remember that the runs are plotted in temporal order).
This is probably because my fitness changed over time, which this model does not account for (we'll get to that later).
Nonetheless, the model is a decent enough representation of the data and I'll use it to calculate GAP estimates.

## Calculating GAP

Armed with this model, I can now calculate my simple GAP estimates by re-arranging my linear model and setting elevation to zero:

$$\log (\mathrm{GAP}) = \log(\mathrm{speed}) + \beta_\mathrm{elevation}\left[\log(\delta) - \log(\mathrm{elevation} + \delta) \right].$$

The counterfactual GAP speed is related to the actual speed through a correction proportional to $$\beta_\mathrm{elevation}$$.
Because I have uncertainty about the value of $$\beta_\mathrm{elevation}$$, I'll also have uncertainty about the GAP value that I infer for each run.
The more elevation there is in a run, the more uncertainty there will be in the estimate of GAP.
To compute this uncertainty, I can just plug my MCMC samples for $$\beta_\mathrm{elevation}$$ into the above formula.
Here's the result of estimating GAP for all of my runs:  

![gap-estimates]({{site.github.url}}/assets/images/strava_post/gap_vs_true.png){:class="img-responsive"}

In this figure, GAP is plotted against my actual pace for the run.
The points are coloured by the elevation gain of the run.
The error-bars are the 95% credible interval for GAP.

The plot broadly makes sense: the GAP estimates are always lower than the true pace (i.e., I would have run faster on the flat), except for the one instance where I did a run where there was zero elevation gain.
Furthermore, runs with more elevation have a bigger difference between GAP and my actual pace.

It's interesting to note that I only did a single run with zero elevation!
This means that I'm leaning on the model assumptions, and hoping that they are plausible when extrapolating to zero elevation gain.
To really test if this is true, I'd need to go out and do some more runs at zero elevation in a variety of conditions and distances (i.e., try to observe something close to the counterfactual).

I couldn't see how to get Strava's own GAP estimate out of the API, so I didn't do a full comparison between the average GAP produced by Strava and my simple model.
I manually grabbed Strava's GAP for the run with the largest elevation gain and did a comparison:  

![me-vs-strava]({{site.github.url}}/assets/images/strava_post/gap_vs_strava.png){:class="img-responsive"}

At least in this case, my approach and the Strava data are consistent with one another.
I probably won't start using this instead of Strava's estimates, but it was good fun to build a simple model myself.

## Modelling my fitness

The simple approach to modelling my running pace produced reasonable results, but we saw in the model checks that there was some autocorrelation in the model errors.
The simple model above does not allow for variation in fitness -- it just says that my pace is a function of how far I'll go and how hilly the run is.
To include some notion of fitness, I expanded the model so that it looks like this:

$$\log\,(\mathrm{pace}_i) =

\alpha_i 
+ \beta_\mathrm{elevation}\log\left(\mathrm{elevation}_i + \delta\right) 
+ \beta_\mathrm{distance}\log\left(\mathrm{distance}_i\right)
+ \epsilon
$$

where the index $$i$$ encodes an ordering to my runs (e.g. run 2 comes after run 1 and before run 3).
The key difference between this model and the original one is that now the intercept $$\alpha$$ is a function of time instead of a constant.
It can be thought of as my "base pace": i.e., fitness.
I should probably index using time explicitly, but I was too lazy for that.
Since I know that my fitness varies smoothly with time, I then put a random walk prior on the intercepts:

$$\alpha_i = \alpha_{i - 1} + \zeta$$,

where $$\zeta$$ is normally distributed noise with variance $$\sigma_\mathrm{rw}^2$$.
The size of the variance controls how rapidly my fitness can change between consecutive runs.
Now, I can infer the set $$\{\alpha_i\}$$ and interpret them as my "fitness" on each of my runs.
Here's the stan code for this model:

```stan
data {
  int n;
  vector[n] log_speed;
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
  log_speed ~ normal(z, sigma);
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

![fitness-trend]({{site.github.url}}/assets/images/strava_post/fitness_trend.png){:class="img-responsive"}

The grey band is the 68% credible interval, and the black dots mark when a run took place.
The results look broadly as I would expect them to. 
I know I was pretty fit last summer, but then had an injury which bothered me until early January.
I then started training again for a couple of half-marathons in May / June.
A definite issue with this approach is that the amount of effort I put into runs is variable, but the model assumes that I am trying my best in every run.
One way to get around this would be to include heart-rate data, which provide a measure of how strained I am during the run.
But as a simple first approach, the results are reasonable.

Let's see if that autocorrelation we saw in the previous model check has been reduced:

![posterior-checks-2]({{site.github.url}}/assets/images/strava_post/residuals.png){:class="img-responsive"}

It definitely has, although there's still some present in the middle of the plot (the large negative residuals clustered together).
I think that this is highlighting an incorrect assumption that I made: fitness varies smoothly over time *unless* you get injured.
Then it changes very abruptly.
I actually got injured last year, and so my runs became notably slower for a while as I recovered.
The injury is effectively a *change-point* in my fitness.
I reckon what's going on is this: because of the random walk prior, the model is forced to smoothly approach a low fitness, which means it underestimates the speed of the runs just before the injury.

## Final thoughts

This was a fun exercise, and it was quite satisfying to do analysis of my own running data!
Very simple statistical models produced relatively interesting insights -- especially the fitness metric in the final section.
I might try adding heart rate data into the model at some point, and see if this improves the results.
If you're interested in trying this for yourself, I put the notebook I used to generate the results from this post in a [github repo](https://github.com/anguswilliams91/negsplit/) (be warned: it's a bit scrappy).