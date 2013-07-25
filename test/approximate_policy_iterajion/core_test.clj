(ns approximate-policy-iterajion.core-test
  (:use clojure.test
        approximate-policy-iterajion.core))

(def test-samples [["b" 5]
                   ["b" 5]
                   ["b" 5]
                   ["b" 5]
                   ["b" 5]
                   ["b" 5]
                   ["b" 5]
                   ["b" 5]
                   ["c" 6]
                   ["d" 7]
                   ["e" 8]
                   ["x" -20]
                   ["z" 20]])

(def statistically-significant? (ns-resolve 'approximate-policy-iterajion.core 'statistically-significant?))
(deftest statistically-singificant-test
  (testing "statistically-significant?")
  (is (= true (statistically-significant? identity 0.05 (range 1 10) 12)))
  (is (= false (statistically-significant? identity 0.05 (range 1 10) 5)))
  (is (= true (statistically-significant? identity 0.1 (range 1 10) 10)))
  (is (= true (statistically-significant? second 0.05 test-samples (last test-samples))))
  (is (= true (statistically-significant? second 0.05 test-samples (nth test-samples 11))))
  (is (= false (statistically-significant? second 0.05 test-samples (nth test-samples 2)))))
