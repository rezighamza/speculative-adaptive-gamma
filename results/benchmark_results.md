### Benchmark Comparison: Fixed vs Adaptive Gamma

| controller    | tok/target-call | accept | mean gamma |
|---------------|-----------------|--------|------------|
| fixed-2       | 2.55            | 0.81   | 2.0        |
| fixed-4       | 3.69            | 0.69   | 4.0        |
| fixed-8       | 5.28            | 0.60   | 8.0        |
| adaptive      | 4.61            | 0.82   | 4.2        |

**Conclusion:** The adaptive controller effectively modulates draft length token-by-token to maximize the tokens generated per target-model call while maintaining a strong acceptance rate (82%), significantly outperforming the efficiency of rigid fixed draft lengths.
