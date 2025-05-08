#import "@preview/equate:0.3.1": equate
#import "@preview/unequivocal-ams:0.1.2": ams-article, theorem, proof

#show: equate.with(breakable: true, sub-numbering: true)
#set math.equation(numbering: "(1.1)")

#let Eps = $cal(E)$
#let eps = $epsilon$
#let nnorm(x) = $norm(size: #50%, #x)$
#let dspace = $space space$
#let tspace = $space space space$
#let nn = $N$
#let ng = $I$
#let nb = $I'$
#let nc = $J$

#let todo(x) = text(red)[*(TODO: #x)*]

#show: ams-article.with(
  title: [Zonotopes],
  authors: (
    (
      name: "Thomas Winninger",
      department: [Sécurité des systèmes et des réseaux],
      organization: [Télécom SudParis],
      location: [Évry],
      email: "thomas.winninger@télécom-sudparis.eu",
      url: "https://sckathach.github.io",
    ),
  ),
  abstract: [Zonotopes are promising abstract domains for machine learning oriented tasks due to their efficiency, but they fail to capture complex transformations needed in new architectures, like the softmax. To overcome this issue, recent work proposed more precise zonotopes, like hybrid constrained zonotopes, or polynomial zonotopes. However, these precise zonotopes often require solving MILP problems at each step, which makes them unusable. This work - focused on the recent transformer architecture - aims to unite different zonotopes methods to reduce the computational overload, while maintaining sufficient precision.],
  bibliography: bibliography("ref.bib"),
)

! _work in progress_ !


= Classical Zonotope

A classical Zonotope @aws_introduction_2021 abstracts a set of $nn in NN$ variables and associates the $k$-th variable with an affine expression $x_k$ using $ng in NN$ noise symbols defined by:

$
  x_k = c_k + sum_(i=1)^ng gamma_(i k) eps_i = c_k + G_k Eps
$

where $c_k, gamma_(i k) in RR$ and $eps_i in [-1, 1]$. The value $x_k$ can deviate from its center coefficient $c_k$ through a series of noise symbols $eps_i$ scaled by the coefficients $gamma_(i k)$. The set of noise symbols $Eps$ is shared among different variables, thus encoding dependencies between $N$ values abstracted by the zonotope.

= Hybrid constrained zonotope
The hybrid constrained zonotope is defined as:
$
  z = c + G Eps + G'Eps', tspace "s.t." space cases(A Eps + A' Eps' = b, Eps in [-1, 1]^ng, Eps' in {-1, 1}^nb)
$

With $nn$ the number of variables, $ng$ the number of continuous noise terms, $nb$ the number of binary noise terms, $nc$ the number of constraints, $z, c in RR^n$, $G in RR^n times RR^ng$, $G' in RR^n times RR^nb$, $A in RR^nc times RR^ng$, $A' in RR^nc times RR^nb$, $b in RR^nc$.

== Concretisation
The lower bound is defined as:
$
  l &= min_(A Eps + A' Eps' = b\ Eps in [-1, 1]^ng\ Eps' in {-1, 1}^nb) c + G Eps + G' Eps'
$
This is a minimisation problem that can be solved with a MILP, which would make compute complexity grow exponentially. However, it is possible to compute sound bounds with relative precision considering the dual problem.

Using Lagrange multipliers, the previous minimisation problem can be rewritten:
$
  l &= min_(Eps in [-1, 1]^ng\ Eps' in {-1, 1}^nb) max_(lambda in RR^nc) c + G Eps + G' Eps' - sum_j^nc lambda_j (A_j Eps + A'_j Eps' - b_j) \
  &= min_(Eps in [-1, 1]^ng\ Eps' in {-1, 1}^nb) max_(Lambda in RR^nn times RR^nc) c + Lambda b - (G - Lambda A) Eps - (G' - Lambda A') Eps'
$

Since the objective is linear and the optimisation variables are compact ($[-1, 1]^ng times {-1, 1}^nb$), we can reverse the order of the min and the max and obtain the same bound:

$
  l &= max_(Lambda in RR^nn times RR^nc) min_(Eps in [-1, 1]^ng\ Eps' in {-1, 1}^nb) c + Lambda b - (G - Lambda A) Eps - (G' - Lambda A') Eps' \
  &= max_(Lambda in RR^nn times RR^nc) c + Lambda b - norm(G - Lambda A)_1 - norm(G' - Lambda A')_1 \
  &= max_(Lambda in RR^nn times RR^nc) d(Lambda)
$
We can observe that, for any $Lambda$, $d(Lambda) <= l$ #todo("Verify"). With $Lambda = 0$, it becomes the concretisation of the classical zonotope. Thus, $d(Lambda)$ is a sound bound that can be optimised.

== Operations @bird_hybrid_2022

For $Z = angle.l c_z, G_z, G'_z, A_z, A'_z, b_z angle.r in RR^nn, Y = angle.l c_y, G_y, G'_y, A_y, A'_y, b_y angle.r in RR^nn, W = angle.l c_w, G_w, G'_w, A_w, A'_w, b_w angle.r in RR^M, R in RR^(M times N)$:

$
  R Z = angle.l R c_z, R G_z, R G'_z, A_z, A'_z, b_z angle.r
$
#let mmat(..a) = $mat(delim: "[", ..#a)$
#let hcz(c, g, gp, a, ap, b) = $lr(angle.l #c, #g, #gp, #a, #ap, #b angle.r)$
#let b0 = $bold(0)$
#let b1 = $bold(1)$
#let bI = $bold(upright(I))$

Minkowski sums:
$
  Z plus.circle Y = hcz(c_z + c_y, mmat(G_z, G_y), mmat(G'_z, G'_y), mmat(A_z, 0; 0, A_y), mmat(A'_z, 0; 0, A'_y), mmat(b_z; b_y))
$

Intersection:
$
  Z inter_R Y = hcz(c_z, mmat(G_z, b0), mmat(G_y, b0), mmat(A_z, b0; b0, A_y; R G_z, -G_y), mmat(A'_z, b0; b0, A'_y; R G'_z, -G'_y), mmat(b_z; b_y; c_y - R c_z))
$

Union, $Z union Y$:
$
  c_u &= 1 / 2 (c_z + c_y + G'_z b1 + G'_y b1) \
  hat(G)' &= 1 / 2 (c_z - c_y + G'_y b1 - G'_z b1) \
  hat(A)'_z &= - 1 / 2 (b_z + A'_z b1), space
  hat(b)_z = 1 / 2 (b_z - A'_z b1) \
  hat(A)'_y &= 1 / 2 (b_y + A'_y b1), space
  hat(b)_y = 1 / 2 (b_y - A'_y b1) \
  G_u &= mmat(G_z, G_y, b0), space
  G'_u = mmat(G'_z, G'_y, hat(G)') \
  A_u &= mmat(
    #grid(
      columns: 3,
      gutter: 7pt,
      [$A_z$], [$b0$], [$b0$],
      [$b0$], [$A_y$], [$b0$],
      grid.cell(colspan: 2, [$A_3$]), [$bI$]
    )
  ), space
  A'_u = mmat(
    #grid(
      columns: 3,
      gutter: 7pt,
      [$A'_z$], [$b0$], [$hat(A)'_z$],
      [$b0$], [$A'_y$], [$hat(A)'_y$],
      grid.cell(colspan: 3, [$A'_3$]),
    )
  ), space
  b_u = mmat(hat(b)_z; hat(b)_y; b_3) \
  A_3 &= mmat(
    bI, b0;
    -bI, b0;
    b0, bI;
    b0, -bI;
    b0, b0;
    b0, b0;
    b0, b0;
    b0, b0;
  ), space
  A'_3 = 1 / 2 mmat(
    b0, b0, b1;
    b0, b0, b1;
    b0, b0, -b1;
    b0, b0, -b1;
    bI, b0, b1;
    -bI, b0, b1;
    b0, bI, -b1;
    b0, -bI, -b1;
  ), space
  b_3 = mmat(
    1 / 2 b1;
    1 / 2 b1;
    1 / 2 b1;
    1 / 2 b1;
    b0;
    b1;
    b0;
    b1;
  ) \
  Z union Y &= hcz(c_u, G_u, G'_u, A_u, A'_u, b_u) subset RR^nn
$


= Abstract Transformers

== General abstract transformer construction
#let upper = $u$
#let lower = $l$
#let tc = $t_"crit"$

@niklas_boosting_2021 provide a general method to find sound and minimal area abstract transformers for zonotopes. Sound neuron-wise transformers for the zonotope domaine can be described as:
$
  y = lambda x + mu + beta eps_"new"
$ <eq:def>

For convex $C^1$ continuous functions, all tangents to the curve of the function yield viable transformers. The resulting parallelogram can be parametrized by the abscissa of the contact point $t$ with $lower ≤ t ≤ upper$. Using the mean value theorem and convexity, it follows that there will be a point $tc$ where the upper edge of the parallelogram will connect the lower and upper endpoints of the graph. For $t < tc$ it will make contact on the upper endpoint and for $t > tc$ on the lower endpoint. This allows to describe the parameters $lambda, mu$ and $beta$ of a zonotope transformer for a function $f(x) : RR -> RR$ on the interval $[lower, upper]$ as:

$
  lambda &= f'(t) \
  mu &= 1 / 2 (f(t) - lambda t + cases(f(lower) - lambda lower \, space "if" t >= tc, f(upper) - lambda upper \, space "if" t < tc)) \
  beta &= 1 / 2 (lambda t - f(t) + cases(f(lower) - lambda lower \, space "if" t >= tc, f(upper) - lambda upper \, space "if" t < tc)) \
  nabla_x f(x)|_(x = tc) &= (f(upper) - f(lower)) / (upper - lower) #<eq:tcrit>
$ <eq:defs>

A minimum area transformer can now be derived by minimizing the looseness $mu$ for $lower <= t <= tc$ and $tc <= t <= upper$. This yields the constrained optimization problems:

$
  &min_t 1 / 2 (f'(t) (t - upper) - f(t) + f(upper)), space space s.t, space lower <= t <= tc #<eq:constraint1> \
  &min_t 1 / 2 (f'(t) (t - lower) - f(t) + f(lower)), space space s.t, space tc <= t <= upper #<eq:constraint2>
$
These can be solved using the method of Lagrange multipliers. @eq:constraint1 leads to the following equations:
#let ll = $cal(L)$
$
  ll &= 1 / 2 (f'(t) (t - upper) - f(t) + f(upper)) + gamma_1 (lower - t) + gamma_2 (t - tc) \
  nabla_t ll &= 1 / 2 f''(t) (t - upper) - gamma_1 + gamma_2 eq 0 \
  nabla_gamma_1 ll &= t - lower \
  nabla_gamma_2 ll &= t - tc \
  gamma_1 &>= 0 \
  gamma_2 &>= 0 \
  gamma_1 (t - l) &= 0 \
  gamma_2 (t - tc) &= 0 \
$

*Case 1:* Neither constraint is active, $gamma_1 = gamma_2 = 0$, $nabla_t ll = f''(t) (t - upper) = 0$. Hence, either $t^* = u = tc$, or $t^*$ verifies $f''(t^*) = 0$.

*Case 2:* $gamma_1 != 0, gamma_2 = 0$, thus $t^* = l$. In this case, $gamma_1 &= 1 / 2 f''(lower) (lower - upper)$. However, as $f$ is convex, $f''(x) >= 0$, so if $u != l$, this leads to $gamma_1 < 0$ which is not possible.

*Case 3:* $gamma_1 = 0, gamma_2 != 0$, thus $t^* = tc$ and $gamma_2 &= 1 / 2 f''(lower) (lower - upper) >= 0$.

*Case 4:* $gamma_1 != 0, gamma_2 != 0$. In this case, $t^* = l = tc$.

Analogously, equation @eq:constraint1 yields a boundary minimum at $t = tc$. Consequently $t=tc$ yields the minimum area transformer for convex functions. $tc$ can be computed either analytically or numerically by solving @eq:tcrit as the point where the local gradient is equal to the mean gradient over the whole interval.

== Exponential Transformer
The exponential function has the feature that its output is always strictly positive, which is important when used as input to the logarithmic function to compute the entropy. Therefore, a guarantee of positivity for the output zonotope is desirable. A constraint yielding such a guarantee can be obtained by inserting $hat(x)_i = lower, eps_(p+1) = - "sign"(mu)$ and $hat(y)_i >= 0$ with $lambda(t) = e^t$ into @eq:def:
#let tc2 = $t_("crit", 2)$
#let to = $t_"opt"$
$
  0 <=& lambda lower + 1 / 2 (f(t) - lambda t + f(upper - lambda upper)) - 1 / 2 (lambda t - f(t) + f(upper - lambda upper)) \
  0 <=& lambda (lower - t) + f(t) \
  0 <=& e^t (lower - t + 1) \
  t <=& 1 + lower eq.triple tc2
$

This constitutes the additional upper limit $tc2$ on $t$. Therefore it is sufficient to reevaluate 16 as it will either be inactive in equation 17 if $tc <= tc2$ for the solutions computed previously or the constraints will be insatiable ensuring that 17 will have no solutions. If a strictly positive output is required a small delta can simply be subtracted from the upper limit $tc2$ . It is easy to see that $t$ is now constrained to $[lower , min(upper, tc2)]$ and that the minimum area solution will be obtained with $to = min(tc, tc2)$. The critical points can be computed explicitly to $tc = log(e^upper − e^lower)$ and $tc2 = lower + 1$. This can be inserted into equations 11 to 14 to obtain a positive, sound and viable transformer.

== Logarithmic Transformer


The logarithmic transformer can be constructed by plugging $f (t) = −log(t)$ and $f'(t) =-1 / x$ into equations 12 to 14 and their results into equation 11. Equation 15 can be solved to $tc = (lower −upper ) / (ln(lower)−ln(upper))$.

// == Affine Abstract Transformer

// The abstract transformer for an affine combination $z = a x_1 + b x_2 + c$ of two Multi-norm Zonotope variables $x_1 = c_1 + aa_1 dot es + bb_1 dot ei$ and $x_2 = c_2 + aa_2 dot es + bb_2 dot ei$, is:

// $
//   z &= a x_1 + b x_2 + c \
//   &= a(c_1 + aa_1 dot es + bb_1 dot ei) + b(c_2 + aa_2 dot es + bb_2 dot ei) + c \
//   &= (a c_1 + b c_2 + c) + (a aa_1 + b aa_2) dot es + (a bb_1 + b bb_2) dot ei
// $
//
// This transformer is exact, as it simply applies the affine operation directly to the Multi-norm Zonotope representation without introducing any over-approximation.

== ReLU Abstract Transformer

The ReLU abstract transformer defined for the classical Zonotope @singh_fast_2018 can be extended naturally to the multi-norm setting @boanert_fast_2021 since it relies only on the lower and upper bounds of the variables, which are computed using the method described for the Multi-norm Zonotope.

For a zonotope variable $x$ with lower bound $l$ and upper bound $u$, the Multi-norm Zonotope abstract transformer for $"ReLU"(x) = max(0, x)$ is:

$
  y = cases(
    0\, &"if" u < 0,
    x\, &"if" l > 0,
    lambda x + mu + beta_"new" eps_"new"\, space space &"otherwise"
  )
$

where $eps_"new" in [-1, 1]$ denotes a new noise symbol, and:

$
  lambda &= u / (u - l) \
  beta_"new" = mu &= 0.5 max(-lambda l, (1 - lambda)u) \
$

We note that the newly introduced noise symbol $eps_"new"$ is an $ell_oo$ noise symbol. This holds for all $eps_"new"$ in the following transformers as well.

#figure(
  grid(
    columns: 2,
    image("assets/relu_mid.svg"), image("assets/relu_right.svg"),
  ),
)

== Tanh Abstract Transformer

The abstract transformer for the operation $y = tanh(x)$ is:

$
  y = lambda x + mu + beta_"new" eps_"new"
$

where:

$
  lambda &= min(1 - tanh^2(l), 1 - tanh^2(u)) \
  mu &= 1 / 2 (tanh(u) + tanh(l) - lambda(u + l)) \
  beta_"new" &= 1 / 2 (tanh(u) - tanh(l) - lambda(u - l))
$

#figure(image("assets/tanh.svg"))

== Exponential Abstract Transformer

The operation $y = e^x$ can be modeled through the element-wise abstract transformer:

$
  y = lambda x + mu + beta_"new" eps_"new"
$

where:

$
  lambda &= e^(t_"opt") \
  mu &= 0.5(e^(t_"opt") - lambda t_"opt" + e^u - lambda u) \
  beta_"new" &= 0.5(lambda t_"opt" - e^(t_"opt") + e^u - lambda u)
$

and

$
  t_"opt" &= min(t_"crit", t_"crit,2") \
  t_"crit" &= log((e^u - e^l) / (u - l)) \
  t_"crit,2" &= l + 1 - hat(eps)
$

Here, $hat(eps)$ is a small positive constant value, such as 0.01. The choice $t_"opt" = min(t_"crit", t_"crit,2")$ ensures that $y$ is positive.

#figure(image("assets/exp_large.svg"))


== Reciprocal Abstract Transformer

The abstract transformer for $y = 1 / x$ with $x > 0$ is given by:

$
  y = lambda x + mu + beta_"new" eps_"new"
$

where:

$
  lambda &= -1 / t_"opt"^2 \
  mu &= 0.5(1 / t_"opt" - lambda dot t_"opt" + 1 / l - lambda l) \
  beta_"new" &= 0.5(lambda dot t_"opt" - 1 / t_"opt" + 1 / l - lambda l)
$

and

$
  t_"opt" &= min(t_"crit", t_"crit,2") \
  t_"crit" &= sqrt(u l) \
  t_"crit,2" &= 0.5 u + hat(eps)
$

Similarly to the exponential transformer, $hat(eps)$ is a small positive constant and $t_"opt" = min(t_"crit", t_"crit,2")$ ensures that $y$ is positive.

#figure(image("assets/reciprocal.svg"))


