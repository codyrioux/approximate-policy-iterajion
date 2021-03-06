# approximate-policy-iterajion

An implementation of Approximate Policy Iteration (API) from the paper Lagoudakis et. al. 2003.

This is a reinforcement learning algorithm that exploits a classifier, in this case an svm, to select
state and action pairs in a large state space.

This algorithm approximates a policy for approximating solutions to very large Markov Decision Processes (MDP)
in a parallel fashion. I plan on using it for part of a larger project, however this component itself is
very reusable so I factored it out into a library.

## Usage

Add the following dependency to your `project.clj` file.

```clojure
[apprpoximate-policy-iterajion "0.4.4"]
```

All of the following code can be found in `sample.clj`

For our toy problem we will be defining a simple problem in which an agent attempts to reach the number 10 using addition.
In this example a state will be a number, and an action will be a number as well (which gets added to the state).

```clojure
(def goal 10)
```

First we need a generative model that will take state and action pairs, and return the new state `sprime` and a reward `r`.
To generate sprime we add `s + a` and to generate a reward we compute `1 / (|goal - (s + a)\) + 0.01)`

```clojure
(defn m
  "States and actions are added."
  [s a]
  (cond
    (nil? a) s
    :else (+ s a))
```

Now a reward function.

```clojure
(defn reward
  [s]
  (/ 1 (+ 0.01 (Math/abs (- goal s)))))
```

Now we need a function to generate a bunch of starting states. For our problem we will start at every number from
0 to 20.

```clojure
(defn dp
  "0 to goal * 2 for starting states"
  [states-1 pi]
  (range 0 (* goal 2)))
```

Now we require a function `sp` that generates actions for a given state. In this example actions available are the same
no matter the state, however in a real world problem actions will vary by state. In this case we will allow the user to
add any number between `-(goal / 2) and (goal / 2)`

```clojure
(defn sp
  "Can add or subtract up to half of the goal."
  [s]
  (cond
   (= goal s) []
   :else (range (* -1 (/ goal 2)) (/ goal 2))))
```

Lastly we require a feature extraction function in order to teach our learner. approximate-policy-iterajion uses svm-clj
under the hood so our features are maps of increasing numbers 1..n to the feature value.

```clojure
(defn features
  [s a]
  {1 s
   2 (- goal s)
   3 (if (pos? s) 1 0)
   4 (if (pos? (- goal s)) 1 0)
   5 (if (> goal s) 1 0)
   6 a
   })
```

Now that we have defined m, dp, sp, and features we can run approximate policy iteration with 300 rollouts per state,
 and a trajectory length of 10 per rollout using a discount factor of 0.99.

```clojure
(use 'approximate-policy-iterajion.core)

(def my-policy (api/api m reward dp sp 0.99 300 10 features "sample" 5 :kernel-type (:rbf api/kernel-types))))

; We get some output from the underlying svm implementation

; Now lets ask the policy for an action given our state s

(my-policy 0)
;=> 4
(my-policy 4)
;=> 4 
(my-policy 8)
;=> 2
(my-policy 10)
;=> nil
```

All of this code is available in `sample.clj` and can be run simply by calling:

```clojure
(use 'approximate-policy-iterajion.sample :reload-all)
(def my-policy (create-api-policy 300 10))
(my-policy 0)
;=> 4
(my-policy 4)
;=> 4 
(my-policy 8)
;=> 2
(my-policy 10)
;=> nil
```

Now take this and build your own reinforcement learning solutions to problems. :D

## Changelog

### 0.4.4
Altered the generated policy so that operations are performed in parallel. The reasoning here
is that should the policy encounter a state it has not seen and need to evaluate many actions
the performance gain is large, whereas if a small number of actions are up for evaluation
the performance loss will be minimal.

### 0.4.3
Altered the code base so that situations in which no action exist can be handled. In this case
the policy functions return nil. Therefore your state generator `sp` can return an empty list
and your generative model `m` should be able to handle nil in place of action.

### 0.4.2
Added a maximum iterations (mi) parameter to api. Allows the user to constrain the run
time of the learner.

### 0.4.1
The deployment to Clojars failed for 0.4.0 so I needed to push a new version.

### 0.4.0
Simplified the API function signature, policy is no longer a parameter. Implements a proper greedy (on estimated value)
returned policy.

### 0.3.11
Another attempt at fixing the divide by zero bug occuring in the t-test that determines
significance.

### 0.3.10
16 agents are now used for parallelism in the application.

### 0.3.9
The bug supposedly fixed in 0.3.7 appears to still exist, and this is another attempt at fixing
said bug.

### 0.3.8
Moved the default policy to one that is random during training and greedy on the estimated (rollout)
values during testing.

### 0.3.7
Fixed a bug in which an exception was thrown if qpi contained only one state-action pair.

### 0.3.6
Added an id parameter to the `api` function. This allows the run to identify itself and persist
and load its training set in case of interruption. Useful for EC2 spot instance computation.

### 0.3.5
Added a branching factor parameter to the `api` function. This allows you to chunk the dataset into the
specified number of pieces for parallel processing. In experimentation the default pmap settings did not
work well. Setting the number to the number of processors in the machine proved much more useful.

### 0.3.4

Removed the pmap from the rollout function. It appears as though any attempt at using the svm model in parallel creates
a resource deadlock. I'll need to explore classifiers in the future that will work for this purpose.

### 0.3.3
The parameter function `dp` is now provided with `states-1` the set of states used in the last iteration and `pi` the policy.
The intention is to allow people to guide their state generation using the policy in the event that the state space is very large.


## Todo
* Unit Tests
* Explore alternate classifiers

## License

Copyright © 2013 Cody Rioux
Distributed under the Eclipse Public License, the same as Clojure.
