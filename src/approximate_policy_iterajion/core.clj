(ns approximate-policy-iterajion.core
  (:require [incanter.stats :as stats])
  (:use [clojure.set]
        [svm.core]))

(defn statistically-significant?
  "Determines if the provided target-sample is statistically significant compared to all samples.  Arguments:
   p-threshold    : The p value threshold for the t-test. ex. 0.05 or 0.01
   score          : A function returning the score of a sample. Note if precomputed in a collection/hash
                    this function could be first or :score for example.
   samples        : A collection of all of the samples.
   target-sample  : The sample for which we would like to determine statistically significant difference.
   
   Examples:
   (statistically-significant? 0.05 identity (range 1 10) 15)"
  [score p-threshold samples target-sample]
  (>= p-threshold (:p-value(stats/t-test 
                             (map score samples) 
                             :mu (score target-sample)))))

(defn- policy
  "Executes the policy on the corresponding state and determines a proper action."
  [sp model state]
  (->>
    (sp state)
    (map #(identity [(predict model state) [state %1]]))
    (filter #(= 1.0 (first %1)))
    (rand-nth)
    (second) 
    (second)))

(defn- calculate-qk
  "Arguments:
   m  : Generative model a function that takes s, a.
   s  : A state, paired with the action.
   a  : An action, paired with the state.
   t  : The length of each trajectory.
   y  : Discount factor. 0 < y <= 1"
  [m s a t y]
  (let [[sprime r] (m s a)]
    (loop [m m, s sprime, a a, t t, qk r, y y]
      (cond 
        (= 0 t) qk
        :else (let [[sprime r] (m s a)]
                (recur m sprime, (policy sprime) (- t 1) (+ qk (* (Math/pow y t) r)) y))))))

(defn rollout
  "This function estimates the value of a state-action pair using rollouts. The underlying concept
   is that the state space for our Markov Decision Process (MDP) is too large to compute exactly.

   Arguments:
   m  : Generative model a function that takes s, a and a seed integer.
   s  : A state, paired with the action.
   a  : An action, paired with the state.
   y  : Discount factor. 0 < y <= 1
   pi : A policy.
   k  : Number of trajectories.
   t  : The length of each trajectory.

   Returns:
   A tuple of the form [[state action] approximated-value]"
  [m, s, a, y, pi, k, t]
  [[s, a] (* (/ 1 k) (apply + (pmap (fn [_] (calculate-qk m s a y)) (range 0 k))))])

(defn- get-positive-samples
  "Takes a series of rollout scores and returns a set containing a single positive training example."
  [samples a*]
  (let [target-sample (first (filter #(= a* (second %1)) samples))
        significant (statistically-significant? second 0.05 samples target-sample)]
    (if significant #{[1 target-sample]} #{})))

(defn- get-negative-samples
  "Takes a series of rollout scores and returns a set containing all the negative training examples."
  [samples]
  (let [sample-mean (/ (reduce + (map second samples)) (count samples))]
    (->>
      (filter #(> sample-mean (second %1)) samples)
      (filter #(statistically-significant? second 0.05 samples %1))
      (map #(identity [-1 %1]))
      (set))))

(defn api
  " The primary function for approximate policy iteration.
   Arguments:
   m  : Generative model.
   dp : A source of rollout states.
   sp : A source of available actions for a state.
   y  : Discount factor.
   pi0: Initial policy.
   k  : The number of trajectories to compute on each rollout.
   t  : The length of each trajectory.

   Returns: A clj-svm model containing a classifier implementing the learned policy."
  [m dp sp y pi0 k t]
  (loop [model nil ts #{} tsi-1 nil]
    (cond
      (= tsi-1 ts) model
      :else (let [qpi (apply concat (for [s (dp)] (for [a (sp s)] (rollout m s a y pi0 k t)))) 
                  a* (max (map second qpi))
                  next-ts (union ts (get-positive-samples qpi) (get-negative-samples qpi)) ]
              (recur (train-model next-ts) next-ts ts)))))
