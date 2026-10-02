### Benchmark Comparison: Fixed vs Adaptive Gamma

| controller    | tok/target-call | accept | mean gamma |
|---------------|-----------------|--------|------------|
| fixed-2       | 1.52            | 0.82   | 2.0        |
| fixed-4       | 2.11            | 0.68   | 4.0        |
| fixed-8       | 2.51            | 0.49   | 8.0        |
| adaptive      | 2.89            | 0.78   | 6.4        |

**Conclusion:** The adaptive controller effectively modulates draft length token-by-token to maximize the tokens generated per target-model call while maintaining a strong acceptance rate.
