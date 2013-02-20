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

(deftest statistically-singificant-test
  (testing "statistically-significant?")
  (is (= true (statistically-significant? identity 0.05 (range 1 10) 12)))
  (is (= false (statistically-significant? identity 0.05 (range 1 10) 5)))
  (is (= true (statistically-significant? identity 0.1 (range 1 10) 10)))
  (is (= true (statistically-significant? second 0.05 test-samples (last test-samples))))
  (is (= true (statistically-significant? second 0.05 test-samples (nth test-samples 11))))
  (is (= false (statistically-significant? second 0.05 test-samples (nth test-samples 2)))))

(def get-positive-samples (ns-resolve 'approximate-policy-iterajion.core 'get-positive-samples))
(deftest get-positive-samples-test
  (testing "get-positive-samples"
    (is (= #{[1 ["z" 20]]} (get-positive-samples test-samples 20)))
    (is (= #{} (get-positive-samples test-samples 5)))))

(def get-negative-samples (ns-resolve 'approximate-policy-iterajion.core 'get-negative-samples))
(deftest get-negative-samples-test
  (testing "get-negative-samples"
    (is (= #{[-1 ["x" -20]]} (get-negative-samples test-samples)))
    (is (= #{} (get-negative-samples (take 5 test-samples))))))
