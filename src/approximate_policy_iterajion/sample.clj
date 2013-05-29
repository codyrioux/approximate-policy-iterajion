(ns approximate-policy-iterajion.sample
  "A sample problem for approximate policy iterajion
   in which we attempt to find the number 100.
   
   States are defined as just a number.
   
   Actions are a number which will be added to the state."
  (:require [approximate-policy-iterajion.core :as api]))

;; Experiment Functions

(defn m
  "States and actions are added.
   Reward is 1 / (|(100 - (s + a))| + 0.01)"
  [s a]
;  (if (or (not (number? s)) (not (number? a))) (println (str "Called with s: " s " and a: " a)))
  [(+ s a)  (/ 1 (+ 0.01 (Math/abs (- 100 (+ s a))))) ])

(defn dp
  "1000 initial states somewhere from 0 to 200"
  []
  (repeat 10 (rand-int 200)))

(defn sp
  "At any state we can add anywhere from -10 to 10.
   This is arbitrarily chosen."
  [s]
  (range -10 11))

(defn features
  "Features are the value of the state and the difference from 100 (the target)"
  [s]
  {1 s, 2 (Math/abs (- 100 s))})

;; Utility Functions

(defn create-api-policy
  "Build a function that implements our policy."
  [trajectory-count trajectory-length]
  (api/api m dp sp 0.5 (partial api/policy features) trajectory-count trajectory-length features))

(defn create-random-policy
  "Build a random policy, for benchmarking our function."
  []
  (fn [x] (rand-nth (sp 1))))

(defn find-100
  "Returns the number of actions necessary to find 100 using policy.
   After 1000 iterations it fails out."
  [policy]
  (loop [state 0 t 0]
    (cond
      (= 100 state) t
      (= t 1000) t
      :else (recur (+ state (policy state)) (+ t 1)))))

(defn run-experiment
  "Tries to find 100 with the policy n times, returning a collection
   containing the number of iterations necessary."
  [policy n]
  (doall (map (fn [x] (find-100 policy)) (range n))))

(defn does-api-beat-random?
  "Returns true if the mean number of iterations for our policy
   is statistically significant in comparison to a random policy."
  []
  (let [api-results (run-experiment (create-api-policy 5 2) 10)
        random-results (run-experiment (create-random-policy) 10)
        api-mean (/ (reduce + api-results) (count api-results))]
    (api/statistically-significant? identity 0.05 random-results api-mean)))
