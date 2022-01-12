# BAA_Optimization
some of my work in internship of second year at Armis

**The src folder: baa_optimization.py contains the code for the optimization algo; input_generator gets the requirements we need in the optimization algo(daily budget,bussiness constraints etc.) ; main.py is the executor of the algo.**

**The resource folder contains the sources of the requirements(strategies and constraints)**

It's a problem of Budget Allocation Optimization(BAA). The BAA aims to minimize the overall CPC (cost per click) by optimizing the daily allocation of the budget between different strategies in different platforms for a campaign, at the beginning of the day.  This is an optimization problem with some constraints. 

For a certain campaign (Monoprix for example), it wants to show its advertisements in different platforms(Facebook, Google, Xandr etc), and for each platform it has some strategies which represent the way of showing the advertisements.  The campaign has a total budget for a period of time (10 days), and at the beginning of each day, the daily budget is calculated by taking into account the remaining budget left and the margin of the previous day. Then the constraints are formed: 

The sum of the budget allocated on each strategy equals to the daily budget 

For each strategy, the allocated budget is ideally less than the spending capability of this strategy, which is estimated from the one of the previous day. 

The allocated budget for any strategy should be no less than the budget we want to spend on it.

In the second constraint, in the case that the allocated budget is larger than the spending capability, we add a relaxation term R in order that the allocated budget is no larger than the sum of the two terms, and we add a penalized term to the term that we want to minimize. 

