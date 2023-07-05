Functions:
- Number of chunks in matrix:
	- $y(x)=x^2$
	- $x$: number of segments on one axis.
	- $y$: number of chunks in entire matrix.
- Number of chunks to load for matrix multiplication of one chunk:
	- $y(x)=2x-1$
	- $x$: number of segments on one axis.
	- $y$: number of chunks to load.
- Number of chunks required for multiplying one chunk:
	- $y(x) = 2x$
	- $x$: number of segments on one axis.
	- $y$: number of chunks loaded.
- Number of chunks to load for multiplying all chunks (naïve/algorithm1):
	- $y(x)=(2x-1)x^2$
	- $x$: number of segments on one axis.
	- $y$: number of chunks to load.
	- Alternative form: $y(x) = 2x^3 - x^2$
- Number of chunks to load for multiplying all chunks (optimised/algorithm2):
	- $y(x)=𝑥^2+(𝑥−1) 𝑥^2$
	- $x$: number of segments on one axis.
	- $y$: number of chunks to load.
	- Alternative form: $y(x) = x^3$
- Algo3:
	- $y(x)=x^2+\sum_{i=1}^{x-1}2(x-1)(x-i)$
	- $x$: number of segments on one axis.
	- $y$: number of chunks to load.
	- Alternative form: $y(x)=𝑥^3−𝑥^2+𝑥$
- Finding optimal chunk size based on matrix size and memory constraints:
	- $2fn-VM=0$ -> $f=\frac{VM}{2n}$
	- $f$: optimal chunk size (bytes).
	- $VM$: maximum memory size (bytes).
	- $n$: size of matrix (bytes).


----
Formula for calculating chunk size based on matrix size, provided by Hans:
$2fn-f^2-VM=0$
Not used and is invalid, $f^2$ is required and should not be subtracted.
