(ns approximate-policy-iterajion.core
  "A reinforcement learning as classification implementation from Lagoudakis et al. 2003.
  
   For each state generated by dp, and each action for that state generated by sp
   performs a rollout, stores each estimated value in qpi. Rollout: [sprime value]
   Then takes the maximum estimated value and stores it in a*.
   Then extracts the positive, negative training examples, unions them with the current ones.
   Terminates when the policy converges, ie. the training set does not change.

   The initial policy is to randomly select an action, this is in effect only on the first iteration
   before training examples are generated.

   Call api with specified parameters to generate a policy function."
  (:require [incanter.stats :as stats]
            [svm.core :as svm])
  (:use [clojure.set]))

(def kernel-types svm/kernel-types)
(def svm-types svm/svm-types)
(def predict svm/predict)

;; ## Helper Functions
;; A set of functions used internally by the algorithm to perform it's calculations.
;; Necessary for a full understanding of the algorithm but not necessary in order to use
;; the algorithm.

(defn- statistically-significant?
  "Determines if the provided target-sample is statistically significant compared to all samples.  Arguments:

   p-threshold: The p value threshold for the t-test. ex. 0.05 or 0.01
   
   score: A function returning the score of a sample. Note if precomputed in a collection/hash
          this function could be first or :score for example.

   samples: A collection of all of the samples.

   target-sample: The sample for which we would like to determine statistically significant difference.

   Examples: `(statistically-significant? identity 0.05 (range 1 10) 15)`"
  [score p-threshold samples target-sample]
  (if (< (count (set (map score samples))) 2)
    true
    (>= p-threshold (:p-value (stats/t-test 
                                (map score samples) 
                                :mu (score target-sample))))))

(defn- get-training-samples
  "Generates the training examples used for the underlying classifier.
   If there is a demonstrably superior action, it is classified as positive and all others as negative.
   If there is not, then the demonstrably inferior examples are classified as negative.

   fe: Feature extractor for [state action] pairs.

   qpi: The samples in the format [[state action] reward]

   Returns a set of classification examples in the form [-1.0 {1 0, 2 0, 3 1...}]."
  [fe qpi]
  (if (< (count qpi) 2)
    #{}
    (let [sample-mean (/ (reduce + (map second qpi)) (count qpi)) 
          amax (apply max (map second qpi))
          qmax (first (filter #(= amax (second %)) qpi))
          a* (if (statistically-significant? second 0.05 qpi qmax) qmax nil)]
      (if a*
        (union #{[1.0 (apply fe (first a*))]} (set (map #(vec [-1.0 (apply fe (first %))]) (filter #(not (= a* %)) qpi))))
        (->>
          (filter #(> sample-mean (second %1)) qpi)
          (filter #(statistically-significant? second 0.05 qpi %1))
          (map #(vec [-1.0 (apply fe (first %1))]))
          (set)))))) 

(defn- rollout
  "Performs value estimations of a state-action pair using rollouts. The underlying concept
   is that the state space for our Markov Decision Process (MDP) is too large to compute exactly.
   The approximated value returned is the estimated value of applying action to state.

   Arguments: Defer to documentation for function `api` for arguments.

   Returns: A tuple of the form `[[state action] approximated-value]`"
  [m, rw, s, a, y, pi, k, t]
  [[s a] (* (/ 1 k) (reduce + (map (fn [_] (let [sprime (m s a) r (rw sprime)]
                                                (loop [s sprime, t t, qk r, y y]
                                                  (cond 
                                                    (= 0 t) qk
                                                    :else (let [sprime (m s (pi s)) r (rw sprime)]
                                                            (recur sprime (- t 1) (+ qk (* (Math/pow y t) r)) y)))))) (range 0 k))))])

(defn- worker
  "Worker thread that can be run in parallel to compute chunks of `[state action]` pairs."
  [work sp pi0 m rw y ts k t options]
  (let [pi (if (= #{} ts) #(rand-nth (sp %)) (partial pi0 (apply svm/train-model (conj options ts))))]
    (doall (map #(rollout m rw (first %) (second %) y pi k t) work))))

;; ## Interface Functions
;; The two user facing functions are `policy` and `api`. Framiliarity with these two functions are necessary to use
;; the algorithm. Essentially running `api` produces a partialed `policy` which you can provide
;; just `state` as a parameter and receive an `action`.

(defn policy
  "Executes the policy on the corresponding state and determines a proper action.
   Generates all possible actions using sp, then maps these actions to their state using m
   It then  classifies each as either positive 1.0 or negative -1.0, placing them in a tuple.

   Note: This is an example policy and you are encouraged to roll your own.

   Arguments: 
   k                 : The number of trajectories to compute for each rollout.

   t                 : The length of each trajectory for each rollout.

   y                 : The discount factor for rollout trajectories.

   feature-extractor : A feature extractor function that takes a state and returns a set of features for the learner.

   rw                : Reward function that takes a state and returns a reward.
   
   sp                : A function that takes a state and returns all possible actions.

   m                 : A generative model that takes `(s, a)` and returns a new state.

   model             : A svm-clj model.

   state             : The current state to use as a base for the next action.

   The algorithm will take all positive samples and attempt to maximize reward.
   If no actions are classified it attempts to maximize reward over all actions.

   Returns: The next action as decided by the policy, randomly selected amongst available options.
   When not in training, it is greedy (on the estimated value) of each action."
  [k t y feature-extractor rw sp m model state & {:keys [mode] :or {mode :training}}]
  (let [actions (filter #(= 1.0 (svm/predict model (feature-extractor state % ))) (sp state))
        actions (if (> (count actions) 0) actions (sp state))]
    (if (= mode :training)
      (rand-nth actions)
      (let [q-actions (zipmap (map #(second (rollout m rw state % y (partial policy k t y feature-extractor rw sp m model) k t)) actions) actions)]
        (q-actions (reduce max (keys q-actions)))))))

(defn api
  "The primary function for approximate policy iteration.

   Arguments:

   m: Generative model, takes `[state action]` pairs and returns a new state.

   rw: Reward function, takes a state and returns a reward value.
   
   dp: A source of rollout states, a fn that returns a set of states. `(dp states-1 pi)`

   sp: A source of available actions for a state, an fn that takes a state and returns actions. `(sp state)`

   y: Discount factor for rollout trajectories. 0 to 1

   pi0: A policy function that takes a model and state, returns an action.

   k: The number of trajectories to compute on each rollout.

   t: The length of each trajectory.

   fe: A function that extracts features from a `[state action]` pair `{1 feature1, 2 feature2, ...}`

   bf: Branching factor, use 1 for completely serial. Be careful, too high and you'll deadlock!

   id: An identifier for this run, used to persist the dataset in case of interruption. 

   options: Options as specified by svm-clj https://github.com/r0man/svm-clj/blob/master/src/svm/core.clj 

   Returns: A function pi that takes state and returns an action."
  [m rw dp sp y pi0 k t fe bf id & options]
  (loop [pi #(rand-nth (sp %))
         ts (let [f (clojure.java.io/file id)
                  ds (if (.exists f) (read-string (slurp id)) #{})]
              ds)
         tsi-1 nil
         states-1 nil]
    (cond
      (= tsi-1 ts) pi
      :else (let [states (dp states-1 pi)
                  state-action-pairs (doall (apply concat (for [s states] (for [a (sp s)] [s a]))))
                  work (partition-all (/ (count state-action-pairs) 16) state-action-pairs)
                  agents (map #(agent % :error-handler #(prn %2))work)
                  _ (doall (map #(send % worker sp pi0 m rw y ts k t options) agents))
                  _ (apply await agents)
                  qpi (apply concat (map deref agents))
                  next-ts  (union ts (get-training-samples fe qpi)) ]
              (spit id next-ts)
              (recur (partial pi0 (apply svm/train-model (conj options next-ts))) next-ts ts states)))))
