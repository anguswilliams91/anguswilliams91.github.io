---
title: "Gradient adjusted pace"
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

A lazy search didn't instantly tell me how GAP is calculated, so I thought it would be fun to try and come up with my own recipe for this metric and compare it to the version that Strava produces.


