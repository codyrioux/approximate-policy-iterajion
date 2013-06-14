(ns approximate-policy-iterajion.sample
  "A sample problem for approximate policy iterajion
   in which we attempt to find the goal number.
   
   States are defined as just a number.
   
   Actions are a number which will be added to the state."
  (:require [approximate-policy-iterajion.core :as api])
  (:gen-class))

(def goal 10)

;; Experiment Functions

(defn reward
  "Reward is 1 / (|(goal - (s + a))| + 0.01)"
  [s] 
  (/ 1 (+ 0.01 (Math/abs (- goal s)))))

(defn m
  "States and actions are added."
  [s a]
  (+ s a))

(defn dp
  "0 to goal * 2 for starting states"
  []
  (range 0 (* goal 2)))

(defn sp
  "Can add or subtract up to half of the goal."
  [s]
  (range (* -1 (/ goal 2)) (/ goal 2)))

(defn features
  "Features are the value of the state and the difference from goal"
  [s]
  {1 s
   2 (- goal s)
   3 (if (pos? s) 1 0)
   4 (if (pos? (- goal s)) 1 0)
   5 (if (> goal s) 1 0)
   })

;; Utility Functions

(defn create-api-policy
  "Build a function that implements our policy.
   The returned function can be called with a state and will recommend an action."
  [trajectory-count trajectory-length]
  (api/api m reward dp sp 0.99 (partial api/policy features reward) trajectory-count trajectory-length features))
