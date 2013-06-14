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
[apprpoximate-policy-iterajion "0.2.1"]
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
  (+ s a))
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
  []
  (range 0 (* goal 2)))
```

Now we require a function `sp` that generates actions for a given state. In this example actions available are the same
no matter the state, however in a real world problem actions will vary by state. In this case we will allow the user to
add any number between `-(goal / 2) and (goal / 2)`

```clojure
(defn sp
  "Can add or subtract up to half of the goal."
  [s]
  (range (* -1 (/ goal 2)) (/ goal 2)))
```

Lastly we require a feature extraction function in order to teach our learner. approximate-policy-iterajion uses svm-clj
under the hood so our features are maps of increasing numbers 1..n to the feature value.

```clojure
(defn features
  "Features are the value of the state and the difference from goal"
  [s]
  {1 s
   2 (- goal s)
   3 (if (pos? s) 1 0)
   4 (if (pos? (- goal s)) 1 0)
   5 (if (> goal s) 1 0)
   })
```

Now that we have defined m, dp, sp, and features we can run approximate policy iteration with 300 rollouts per state,
 and a trajectory length of 5 per rollout using a discount factor of 0.99.

```clojure
(use 'approximate-policy-iterajion.core)

(def my-policy (api/api m reward dp sp 0.99 (partial api/policy features reward) 300 10 features)))

; We get some output from the underlying svm implementation

; Now lets ask the policy for an action given our state s

(my-policy 0)
;=> 4
(my-policy 4)
;=> 4 
(my-policy 8)
;=> 2
(my-policy 10)
;=> 0
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
;=> 0
```

Now take this and build your own reinforcement learning solutions to problems. :D

## Todo
* Unit Tests
* Agents for parallelism
* Explore using deep belief networks

## License

Copyright © 2013 Cody Rioux
Distributed under the Eclipse Public License, the same as Clojure.
