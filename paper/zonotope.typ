#import "@preview/codly:1.3.0": *
#import "@preview/codly-languages:0.1.1": *
#show: codly-init.with()

#codly(languages: codly-languages)

#set heading(numbering: "1.")

#let Eps = $cal(E)$
#let eps = $epsilon$
#let phi = $phi.alt$
#let aa = $arrow(alpha)$
#let bb = $arrow(beta)$
#let ei = $arrow(epsilon)$
#let es = $arrow(phi)$
#let nnorm(x) = $norm(size: #50%, #x)$
#let dspace = $space space$

= Multi-Norm Zonotopes


== Classical Zonotope

A classical Zonotope @aws_introduction_2021 abstracts a set of $N in NN$ variables and associates the $k$-th variable with an affine expression $x_k$ using $Eps in NN$ noise symbols defined by:

$
  x_k = c_k + sum_(i=1)^Eps beta^i_k eps_i = c_k + bb_k dot ei
$

where $c_k, beta^i_k in RR$ and $eps_i in [-1, 1]$. The value $x_k$ can deviate from its center coefficient $c_k$ through a series of noise symbols $eps_i$ scaled by the coefficients $beta^i_k$. The set of noise symbols $ei$ is shared among different variables, thus encoding dependencies between $N$ values abstracted by the zonotope.

== Multi-Norm Zonotope Definition

The Multi-norm Zonotope domain @boanert_fast_2021 extends the classical Zonotope by adding noise symbols $phi_j$ that fulfill the constraint $nnorm(es)_p <= 1$, where $es := (phi_1, dots, phi_(Eps_p))^T$. If $p = oo$, we recover the classical Zonotope. This new domain allows us to easily express $ell_p$-norm bound balls in terms of the new noise symbols $phi$:

$
x_k = c_k + sum_(i=1)^(Eps_p) alpha^i_k phi_i + sum_(j=1)^(Eps_oo) beta^j_k eps_j = c_k + aa_k dot es + bb_k dot ei
$

where $c_k, alpha^i_k, beta^j_k in RR$, $nnorm(es)_p <= 1$, and $eps_j in [-1, 1]$.

In the implementation, constants are stored as properties: 
- `z.p: int` - Special norm, $p$ 
- `z.q: int` - Dual norm of $p$ (see @sec:dual-norm), $q$
- `z.Ei: int` - Number of infinity norm error terms, $Eps_p$
- `z.Es: int` - Number of special norm error terms, $Eps_oo$
- `z.N: int` - Number of zonotope variables, $N$

_The properties are dynamic, for instance `z.Ei` will give the current $Eps_oo$, as it may change during operations._

#figure(
  image("assets/example_zonotope.svg"), 
  caption: [A multi-norm Zonotope with two variables $x = 4 + phi_1 - eps_1 + 2 eps_2$, and $y = 3 + phi_2 + eps_1 + eps_2$, where $nnorm(es)_2<=1$ and $eps_1, eps_2 in [-1, 1]$. The green region indicates the classical Zonotope obtained by removing the $es$ noise symbols.]
)

== Matrix Representation

To concisely represent the $N$ zonotope output variables $x_1, dots, x_N$, we write $x = (x_1, dots, x_N)^T$. Therefore, the Multi-norm Zonotope $x$ can be simplified to:

$
x = c + A es + B ei \
c in RR^N, A in RR^(N times Eps_p), B in RR^(N times Eps_oo) \
nnorm(es)_p <= 1, eps_j in [-1, 1]
$

where $A_(k,i) = alpha^i_k$ and $B_(k,j) = beta^j_k$.

In the implementation, error terms are stored in different tensors: 
-  `z.W_Es: Float[Tensor, "N Es"]` - Special error terms, $A in RR^(N times Eps_p)$ 
-  `z.W_Ei: Float[Tensor, "N Ei"]` - Infinity error terms, $B in RR^(N times Eps_oo)$
- `z.W_C: Float[Tensor, "N"]` - Bias or center terms, $c in RR^N$ 



== Dual Norm <sec:dual-norm>

For a given vector $z in RR^N$, the dual norm $nnorm(z)^*_p$ of the $ell_p$ norm is defined as:

$
nnorm(z)^*_p = sup{z dot x | x in RR^N, nnorm(x)_p <= 1}
$

The dual norm $nnorm(z)^*_p$ is the $ell_q$ norm where $q$ satisfies the relationship $1/p + 1/q = 1$.

== Computing Concrete Bounds - `z.concretize()`
The tight lower and upper bounds of $z dot x$ where $x in RR^N$ s.t. $nnorm(x)_p <= 1$ are given by:

$
  l^q_k = -nnorm(z)_q \
  u^q_k = nnorm(z)_q
$

Thus the lower and upper interval bounds of the special terms of the zonotopes can be computed as:  

$
  -nnorm(aa_k)_q <= aa_k dot es <= nnorm(aa_k)_q
$

Given this, the full lower and upper bounds of the multi-norm Zonotope, $l_k$ and $u_k$ of $x_k$ are:

$
  l_k = c_k - nnorm(aa_k)_q + min(bb_k dot ei) = c_k - nnorm(aa_k)_q - nnorm(bb_k)_1 \
  u_k = c_k + nnorm(aa_k)_q + max(bb_k dot ei) = c_k + nnorm(aa_k)_q + nnorm(bb_k)_1
$

== Sampling a Point from Multi-norm Zonotope - `z.sample_point()`

The sampling procedure consists of two parts. We first sampling the $ell_p$-norm noise symbols by generating points within the $ell_p$-norm unit ball. We then sample the infinity norm noise symbols by generating values in $[-1, 1]$. 

=== Sampling the $ell_p$-norm Noise Symbols

To generate a point within the $ell_p$-norm unit ball:

1. Generate a random vector $v in RR^(Eps_p)$ following a standard normal distribution
2. Normalize $v$ by its $p$-norm: $v' = v / nnorm(v)_p$
3. Scale $v'$ by a random factor $r in [0, 1]$ to ensure coverage of the interior of the ball: $es = r dot v'$

This procedure gives us a random point $es$ such that $nnorm(es)_p <= 1$.

=== Sampling the $ell_oo$-norm Noise Symbols

For each $ell_oo$-norm noise symbol $eps_j$, we simply sample a uniform random value in $[-1, 1]$:

$
ei_j tilde "Uniform"(-1, 1)
$

== Noise Symbol Reduction - `z.remove_infinity_errors()`

Through the repeated application of abstract transformers during verification, the number of $ell_oo$ noise symbols grows, leading to slower verification and higher memory usage. In fact, every abstract transformer we use, except the one for affine transformations, can yield new noise symbols $eps_("new")$. To address this, we periodically reduce the number of $ell_oo$ noise symbols to ensure an upper bound on the memory usage independent of the network depth, thus creating a tunable tradeoff between precision and speed.

=== $"DecorrelateMin"_k$ Method

We follow the $"DecorrelateMin"_k$ heuristic method @mirman_provable_2020, that reduces the number of $ell_oo$ noise symbols in a Multi-norm Zonotope to $k$. The method works as follows:

1. *Score Calculation*: For each $ell_oo$ noise symbol $eps_j$, we calculate a score $m_j$ representing its significance:
   
   $
   m_j = sum_(i=1)^N |B_(i,j)| = sum_(i=1)^N |beta^j_i|
   $
   
   where $B_(i,j) = beta^j_i$ is the coefficient of the $j$-th noise symbol for the $i$-th variable.

2. *Ranking*: We rank the noise symbols based on their scores and select the top $k$ noise symbols to keep.

3. *Reduction*: We combine the effects of the eliminated noise symbols into a single new noise symbol for each zonotope variable.

Let $I$ denote the indices of the eliminated $ell_oo$ noise symbols and $P$ the indices of the top $k$ $ell_oo$ noise symbols. Then, the new Multi-norm Zonotope is:

$
x = c + A es + B_P ei_P + [eps_("new",1) sum_(j in I) |beta^j_1|, dots, eps_("new",N) sum_(j in I) |beta^j_N|]
$

Where:
- $ei_P$ represents the kept noise symbols with indices in $P$
- $B_P$ contains only the corresponding columns of $B$
- $eps_("new",i) in [-1, 1]$ for each $i in {1, 2, dots, N}$ are new $ell_oo$ noise symbols

The coefficient for each new noise symbol $eps_("new",i)$ is computed as the sum of absolute values of the coefficients of the eliminated noise symbols for the $i$-th variable:

$
sum_(j in I) |beta^j_i|
$

=== Implementation in the Verification Process

During verification of Transformer networks:

1. The noise symbol reduction is applied to the input embeddings of every Transformer layer, just before the residual connection around the multi-head self-attention.

2. This strategic placement avoids the complexities of handling separate noise reductions in the two branches of the residual connection.

3. The number of retained symbols $k$ can be adjusted based on the available memory and desired precision. A larger $k$ preserves more precision but requires more memory.

This approach effectively controls the growth of noise symbols throughout the network propagation process, ensuring that memory usage remains bounded regardless of network depth, while minimizing precision loss by keeping the most significant noise symbols.

The $"DecorrelateMin"_k$ method creates a favorable balance between verification accuracy and computational efficiency, making it possible to verify deeper networks that would otherwise be intractable.

= Abstract Transformers

== Affine Abstract Transformer

The abstract transformer for an affine combination $z = a x_1 + b x_2 + c$ of two Multi-norm Zonotope variables $x_1 = c_1 + aa_1 dot es + bb_1 dot ei$ and $x_2 = c_2 + aa_2 dot es + bb_2 dot ei$, is:

$
z = a x_1 + b x_2 + c \
= a(c_1 + aa_1 dot es + bb_1 dot ei) + b(c_2 + aa_2 dot es + bb_2 dot ei) + c \
= (a c_1 + b c_2 + c) + (a aa_1 + b aa_2) dot es + (a bb_1 + b bb_2) dot ei
$

This transformer is exact, as it simply applies the affine operation directly to the Multi-norm Zonotope representation without introducing any over-approximation.

== ReLU Abstract Transformer

The ReLU abstract transformer defined for the classical Zonotope @singh_fast_2018 can be extended naturally to the multi-norm setting @boanert_fast_2021 since it relies only on the lower and upper bounds of the variables, which are computed using the method described for the Multi-norm Zonotope. 

For a zonotope variable $x$ with lower bound $l$ and upper bound $u$, the Multi-norm Zonotope abstract transformer for $"ReLU"(x) = max(0, x)$ is:

$
y = cases(
  0, "if" u < 0,
  x, "if" l > 0,
  lambda x + mu + beta_"new" eps_"new", "otherwise"
)
$

where $eps_"new" in [-1, 1]$ denotes a new noise symbol, and:

$
lambda = u / (u - l) \
mu = 0.5 max(-lambda l, (1 - lambda)u) \
beta_"new" = 0.5 max(-lambda l, (1 - lambda)u)
$

We note that the newly introduced noise symbol $eps_"new"$ is an $ell_oo$ noise symbol. This holds for all $eps_"new"$ in the following transformers as well.

The bounds $l$ and $u$ for a Multi-norm Zonotope variable $x_k = c_k + aa_k dot es + bb_k dot ei$ are computed as:

$
l_k = c_k - nnorm(aa_k)_q + min(bb_k dot ei) = c_k - nnorm(aa_k)_q - nnorm(bb_k)_1 \
u_k = c_k + nnorm(aa_k)_q + max(bb_k dot ei) = c_k + nnorm(aa_k)_q + nnorm(bb_k)_1
$

where $q$ satisfies the relationship $1/p + 1/q = 1$ with $p$ being the norm used for $es$.

== Tanh Abstract Transformer

To support the tanh function present in the pooling layer, we extend the abstract transformer for the tanh to the multi-norm setting. As with ReLU, the abstract transformer is applied element-wise. The abstract transformer for the operation $y = tanh(x)$ is:

$
y = lambda x + mu + beta_"new" eps_"new"
$

where $eps_"new" in [-1, 1]$ denotes a new noise symbol and:

$
lambda = min(1 - tanh^2(l), 1 - tanh^2(u)) \
mu = 1/2 (tanh(u) + tanh(l) - lambda(u + l)) \
beta_"new" = 1/2 (tanh(u) - tanh(l) - lambda(u - l))
$

The bounds $l$ and $u$ are computed as in the ReLU transformer.

== Exponential Abstract Transformer

Since the softmax function $sigma$ requires the exponential, we define the abstract transformer for the exponential function that minimizes the area of the output zonotope and ensures its lower bound is positive, which is required by the reciprocal abstract transformer that will be applied afterward.

The operation $y = e^x$ can be modeled through the element-wise abstract transformer:

$
y = lambda x + mu + beta_"new" eps_"new"
$

where $eps_"new" in [-1, 1]$ denotes a new noise symbol, and:

$
lambda = e^(t_"opt") \
mu = 0.5(e^(t_"opt") - lambda t_"opt" + e^u - lambda u) \
beta_"new" = 0.5(lambda t_"opt" - e^(t_"opt") + e^u - lambda u)
$

and

$
t_"opt" = min(t_"crit", t_"crit,2") \
t_"crit" = log((e^u - e^l)/(u - l)) \
t_"crit,2" = l + 1 - hat(eps)
$

Here, $hat(eps)$ is a small positive constant value, such as 0.01. The choice $t_"opt" = min(t_"crit", t_"crit,2")$ ensures that $y$ is positive.

== Reciprocal Abstract Transformer

To obtain the minimal area abstract transformer for the reciprocal function, which is required for the softmax, we use a similar approach to the exponential transformer. The abstract transformer for $y = 1/x$ with $x > 0$ is given by:

$
y = lambda x + mu + beta_"new" eps_"new"
$

where $eps_"new" in [-1, 1]$ denotes a new noise symbol and:

$
lambda = -1/(t_"opt")^2 \
mu = 0.5(1/t_"opt" - lambda dot t_"opt" + 1/l - lambda l) \
beta_"new" = 0.5(lambda dot t_"opt" - 1/t_"opt" + 1/l - lambda l)
$

and

$
t_"opt" = min(t_"crit", t_"crit,2") \
t_"crit" = sqrt(u l) \
t_"crit,2" = 0.5 u + hat(eps)
$

Similarly to the exponential transformer, $hat(eps)$ is a small positive constant and $t_"opt" = min(t_"crit", t_"crit,2")$ ensures that $y$ is positive.

== Dot Product Abstract Transformer

Next, we define the abstract transformer for the dot product between pairs of vectors of variables of a Multi-norm Zonotope. The transformer is used in the multi-head self-attention, specifically in the matrix multiplications between $Q$ and $K$ and between the result of the softmax and $V$.

For two Multi-norm Zonotope vectors $vec(v)_1 = vec(c)_1 + A_1 es + B_1 ei$ and $vec(v)_2 = vec(c)_2 + A_2 es + B_2 ei$, computing the dot product produces the output variable $y$:

$
y = vec(v)_1 dot vec(v)_2 = (vec(c)_1 + A_1 es + B_1 ei) dot (vec(c)_2 + A_2 es + B_2 ei) \
= vec(c)_1 dot vec(c)_2 + (vec(c)_1^T A_2 + vec(c)_2^T A_1) es + (vec(c)_1^T B_2 + vec(c)_2^T B_1) ei \
+ (A_1 es + B_1 ei) dot (A_2 es + B_2 ei)
$

The last term represents interactions between noise symbols and is not in the functional form of a Multi-norm Zonotope. We first expand it:

$
(A_1 es + B_1 ei) dot (A_2 es + B_2 ei) = (A_1 es) dot (A_2 es) + (A_1 es) dot (B_2 ei) \
+ (B_1 ei) dot (A_2 es) + (B_1 ei) dot (B_2 ei)
$

Each of these 4 terms contains a different combination of noise symbols and coefficients. We calculate interval bounds for each combination, e.g., $[l_{es,ei}, u_{es,ei}]$ for $(A_1 es) dot (B_2 ei)$. Then the sum of the lower and upper bounds:

$
l = l_{es,es} + l_{es,ei} + l_{ei,es} + l_{ei,ei} \
u = u_{es,es} + u_{es,ei} + u_{ei,es} + u_{ei,ei}
$

bounds the whole term $l ≤ (A_1 es + B_1 ei) dot (A_2 es + B_2 ei) ≤ u$.

=== Fast Bounds $l_gamma,delta, u_gamma,delta$ (DeepT-Fast)

To compute bounds for a generic expression $(V xi_(p_1)) dot (W xi_(p_2))$, where $V$ and $W$ are matrices such that $V xi_(p_1)$ and $W xi_(p_2)$ have the same dimension and $nnorm(xi_(p_1))_(p_1) ≤ 1$ and $nnorm(xi_(p_2))_(p_2) ≤ 1$, we first compute an upper bound for the absolute value:

$
|(V xi_(p_1)) dot (W xi_(p_2))| = |xi_(p_1)^T V^T W xi_(p_2)| ≤ |xi_(p_1)^T V^T| |W xi_(p_2)|
$

Using Lemma 1, we can bound the elements $|vec(w)_j dot xi_(p_2)|$ of the vector $|W xi_(p_2)|$, where $vec(w)_j$ denotes the $j$-th row of $W$ and $ell_(q_2)$ is the dual norm of $ell_(p_2)$:

$
|(V xi_(p_1)) dot (W xi_(p_2))| ≤ |xi_(p_1)^T V^T| mat(|vec(w)_1 dot xi_(p_2)|, dots, |vec(w)_N dot xi_(p_2)|) \
≤ |xi_(p_1)^T V^T| mat(nnorm(vec(w)_1)_(q_2), dots, nnorm(vec(w)_N)_(q_2)) \
= mat(nnorm(vec(w)_1)_(q_2), dots, nnorm(vec(w)_N)_(q_2))^T |V xi_(p_1)| \
≤ mat(nnorm(vec(w)_1)_(q_2), dots, nnorm(vec(w)_N)_(q_2))^T |V| |xi_(p_1)| \
≤ nnorm(mat(nnorm(vec(w)_1)_(q_2), dots, nnorm(vec(w)_N)_(q_2))^T |V|)_(q_1)
$

where $ell_(q_1)$ is the dual norm of $ell_(p_1)$.

The complexity to compute this bound is $O(N(Eps_p + Eps_oo))$.

=== More Precise Bounds $l_{ei,ei}, u_{ei,ei}$ (DeepT-Precise)

In the case where our Multi-norm Zonotope has solely $ell_oo$ noise symbols (i.e., $p_1 = p_2 = oo$), meaning $xi_(p_1) = xi_(p_2) = ei$, a tighter approximation using interval analysis can be achieved at the cost of increasing the computational complexity to $O(N Eps_oo^2)$.

We begin by summing coefficients related to each pair of noise symbols:

$
(V ei) dot (W ei) = sum_(i=1)^(Eps_oo) sum_(j=1)^(Eps_oo) (vec(v)_i dot vec(w)_j) eps_i eps_j
$

where $vec(v)_i$ and $vec(w)_j$ denote the $i$-th and $j$-th column of $V$ and $W$, respectively. We separate $eps_i^2$ and $eps_i eps_j$ to arrive at:

$
(V ei) dot (W ei) = sum_(i=1)^(Eps_oo) (vec(v)_i dot vec(w)_i) eps_i^2 + sum_(i!=j) (vec(v)_i dot vec(w)_j) eps_i eps_j
$

Since $eps_i^2 in [0, 1]$ and $eps_i eps_j in [-1, 1]$, we have:

$
(V ei) dot (W ei) in sum_(i=1)^(Eps_oo) (vec(v)_i dot vec(w)_i) [0, 1] + sum_(i!=j) (vec(v)_i dot vec(w)_j) [-1, 1]
$

Using interval analysis, we can calculate the lower and upper interval bounds $l_{ei,ei}$ and $u_{ei,ei}$.

The final dot product transformer can combine DeepT-Fast for computing bounds for the mixed terms $(l_{es,es}, u_{es,es})$, $(l_{es,ei}, u_{es,ei})$, and $(l_{ei,es}, u_{ei,es})$, while using DeepT-Precise for computing $(l_{ei,ei}, u_{ei,ei})$.



== Softmax Sum Zonotope Refinement


@ghorbal_logical_2010

By construction, the outputs $y_1, dots, y_N$ of the softmax function $sigma$ when applied to inputs $x_1, dots, x_N$ satisfy $sum_(i=1)^N y_i = 1$, meaning they form a probability distribution. Thus, in the multi-head self-attention, the role of the softmax is to pick some convex combination of the values $V$, according to the similarity between the query and the keys.

However, this property is not always satisfied for the Multi-norm Zonotope obtained for $Z$ produced by the softmax abstract transformer (Eq. 1). By abuse of notation, we call this Zonotope $Z$. There are many valid instantiations of the noise symbols such that the Zonotope variables do not sum to 1, causing non-convex combinations of values to be picked. To address this, we enforce the constraint that the variables must sum to 1, to ensure that a convex combination is selected and to preserve the semantics of the network in our abstract domain. This is achieved by excluding from the Multi-norm Zonotope $Z$ all invalid instantiations of values, obtaining a refined Multi-norm Zonotope $Z'$ with lower volume, that helps to increase verification precision.

We leverage Zonotope constraint methods, which produce refined Zonotopes given some equality constraints. A three-step process is used to refine all Zonotope variables $y_1, dots, y_N$ by:

1. Computing a refined variable $y'_1$ by imposing the equality constraint $y_1 = 1 - (y_2 + dots + y_N)$,
2. Refining all other variables $y_2, dots, y_N$ to $y'_2, dots, y'_N$,
3. Tightening the bounds of the $eps_i$'s to a subset of $[-1, 1]$.

Note that we arbitrarily select $y_1$ as the variable to be refined first, but any other variable could have been chosen.

We now detail these three steps that lead to a refined Zonotope $Z'$ with variables $y'_1, dots, y'_n$ that always sum to 1 and have tighter error bounds.

=== Step 1. Refining $y_1$

We illustrate the process of obtaining a refined Zonotope with variable $z'_1$, given the equality constraint $z_1 = z_2$ for a Zonotope with two variables $z_1$ and $z_2$. The final result can then be obtained by instantiating $z_2 = 1 - (y_2 + dots + y_N)$ and $z_1 = y_1$ and finally $y'_1 = z'_1$.

While we know that $z_1 = z_2$ needs to hold, not all instantiations of the noise symbols satisfy this constraint. We can compute a new Multi-norm Zonotope variable $z'_1 = c' + aa' dot es + bb' dot ei$, such that for all instantiations of noise symbols of $z'_1$, we have $z_1 = z_2$ and $z_1 = z'_1$, thereby enforcing the equality constraints $z_1 = z_2$. We have:

$
z_1 := c_1 + aa_1 dot es + bb_1 dot ei = c_2 + aa_2 dot es + bb_2 dot ei =: z_2
$

If we solve for $eps_k$ (any $k$ such that $beta^k_1 - beta^k_2 != 0$ works) in the equation above and substitute it in the equation $z_2 = z'_1$, we obtain the following constraints for the coefficients of $z'_1$:

$
c' = c_2 + (c_2 - c_1) (beta'^k - beta^k_2) / (beta^k_2 - beta^k_1)
$

$
aa' = aa_2 + (aa_2 - aa_1) (beta'^k - beta^k_2) / (beta^k_2 - beta^k_1)
$

$
bb'^I = bb^I_2 + (bb^I_2 - bb^I_1) (beta'^k - beta^k_2) / (beta^k_2 - beta^k_1)
$

where $I$ are the indices of the other $eps$ terms (i.e., without $eps_k$).

=== Choosing a Value for $beta'^k$

In the equations above, we have one degree of freedom, namely $beta'^k$. Any value $v$ for $beta'^k$ is valid and leads to a valid affine expression $z'_v$, with the other coefficients of $z'_v$ being deduced through the equations above.

To select $v$, we opt to minimize the absolute value of the noise symbol coefficients, which acts as a heuristic for the tightness of the zonotope variable:

$
v^* = min_v S = min_v [nnorm(aa')_1 + nnorm(bb')_1]
$

The minimization problem above can be efficiently solved with $O((Eps_p + Eps_oo) log(Eps_p + Eps_oo))$ complexity, using a method that relies on two observations:

1. Since all coefficients in the minimization can be written in form $r + s beta^k_z$ with $r, s in RR$ (see Eqs. above), the expression to be minimized is of the form $S = sum_t |r_t + s_t beta^k_z|$. The optimal value $v^*$ for $beta^k_z$ will cause one of the $|r_t + s_t beta^k_z|$ terms to be 0 and therefore $v^*$ must equal $-r_i/s_i$ for some $i in [1, Eps_p + Eps_oo]$. The values $-r_i/s_i$ are the candidate solutions for the minimization problem.

2. Each term $|r_t + s_t beta^k_z|$ of $S$ has a constant negative slope before $-r_t/s_t$ and a constant positive slope after it. Therefore, as $beta^k_z$ increases, the slope of more and more $|r_t + s_t beta^k_z|$ terms becomes positive, showing that the slope of $S$ increases monotonically with $beta^k_z$. The minimum value of $S$ will happen at the value of $beta^k_z$ where the slope of $S$ changes from negative to positive.

Since the slope of $S$ increases monotonically, we can run a binary search on $beta^k_z$ to efficiently find the value at which the slope of $S$ changes sign. We note that to maintain precision, we disallow solutions that lead to the elimination of one of the $ell_p$-norm noise symbols $phi$.

=== Step 2. Refining $y_2, dots, y_n$

We substitute the expression for $eps_k$ computed in Step 1 in the affine expressions of the variables $y_2, dots, y_N$, to obtain the refined Multi-norm Zonotope variables $y'_2, dots, y'_N$.

Specifically, from Step 1, we derived:

$
eps_k = (c_1 - c_2 + (aa_1 - aa_2) dot es + sum_(j != k) (beta^j_1 - beta^j_2) eps_j) / (beta^k_2 - beta^k_1)
$

For each variable $y_i$ with $i in {2, dots, N}$, defined as $y_i = c_i + aa_i dot es + bb_i dot ei$, we substitute the expression for $eps_k$ to get:

$
y'_i = c_i + aa_i dot es + sum_(j != k) beta^j_i eps_j + beta^k_i eps_k
$

$
y'_i = c_i + aa_i dot es + sum_(j != k) beta^j_i eps_j + beta^k_i ((c_1 - c_2 + (aa_1 - aa_2) dot es + sum_(j != k) (beta^j_1 - beta^j_2) eps_j) / (beta^k_2 - beta^k_1))
$

After simplification, this gives us the coefficients for our refined variables $y'_i$.

=== Step 3. Tightening the Bounds of $ei$

The refined sum constraint $S = 1 - sum_(i=1)^N y'_i = c_S + aa_S dot es + bb_S dot ei = 0$ can be further leveraged to tighten the bounds of the $ell_oo$ noise symbols $ei$, with non-zero coefficient, by solving for $eps_m$:

$
eps_m = 1/beta^m_S [-c_S - aa_S dot es - bb^I_S dot ei^I]
$

Which implies that the range of $eps_m$ is restricted to $[a_m, b_m] sect [-1, 1]$ where:

$
a_m = 1/|beta^m_S| [c_S - nnorm(aa_S)_q - nnorm(bb_S)_1]
$

$
b_m = 1/|beta^m_S| [c_S + nnorm(aa_S)_q + nnorm(bb_S)_1]
$

Note that because the noise symbol reduction process assumes all noise symbols $ei$ have range $[-1, 1]$, prior to it a pre-processing step occurs where all noise symbols $eps_m$ with tightened bounds $[a_m, b_m] subset [-1, 1]$ are re-written as:

$
eps_m = (a_m + b_m)/2 + (b_m - a_m)/2 eps_("new",m)
$

with $eps_("new",m) in [-1, 1]$.

This three-step refinement process ensures that our Multi-norm Zonotope respects the semantics of the softmax function, improving the precision of our verification procedure.


#bibliography(title: "References", "ref.bib")