Section 1

The path estimated through dead reckoning aligns with resepect to the ground truth 
with small shifts in various segments, more prominent when the turning rate increases
significantly. Moreover, the errors in heading and position do not grow without bounds due to
having the true control inputs at our disposal, and change in magnitude based on the control inputs 
at those particular time frames. Error values settle down to a reasonable magnitude considering 
that we are purely estimating with kinematical model without accounting for dynamics
and other uncertainty factors such as wheel slippage.

Section 2

A series of paths generated with noisy inputs are plotted on Figure 2, depicting how error grows
in the simulation with time. Controls used for every path are susceptible to a uniform noise that differs
from trial to trial, implying that when the noise component is larger in magnitude a path diverges
by a larger magnitude and faster when simulated.

Section 3

The map generated with noisy odometry does not align properly with the ground truth. We can observe
drift due to large control inputs in terms of veocity since we are not mapping when the angular rates
exceed 0.1 rad/s - displaced regions due to uncertainty accumulated throughout driving and dead
reckoning. Moreover, several map segments are duplicated and displacd from one another due to drift,
dead reckoning accumulated error, and scanning of the same region between consecutive time stamps to another.
Of course, errors due to interpolation of the noisy odometry estimations are additional factors for the observed
inaccuracies between the true map and the estimated one.
 