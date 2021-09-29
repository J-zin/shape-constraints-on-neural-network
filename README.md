# Shape Constraints on Nerual Networks

This repo containts the shape constraints methods based on *ordinary differential equation*, which is first proposed in [UMNN (Unconstrainted Monontonic Neural Network)](https://arxiv.org/abs/1908.05164). In UMNN, the authors conduct monotonic constraint on neural network by parameterzing the function as
$$
F(x;\psi) = \int_0^x f(t; \psi) \mathrm{d} t + F(0;\psi), \\
f(x;\psi) =: \frac{\partial F(x;\psi)}{\partial x} > 0.
$$
Here, we extend this framework into another shape constraints beyond monotone, i.e., **increasing concave** and **general concave** functions.

**increasing concave**:
$$
F(x;\psi) = \int_{a=0}^{a=x} \int_{b=a}^{b=\infty} f(b; \psi) \mathrm{d} b \mathrm{d} a\\
\nabla_x F(x;\psi) = \int_{b=x}^{b=\infty} f(b; \psi) \mathrm{d}b > 0 \\
\nabla_x^2 F(x;\psi) = -f(x; \psi).
$$
**general concave**:
$$
F(x;\psi) = \int_{a=0}^{a=x} \int_{b=a}^{b=\infty} f^+ (b; \psi) \mathrm{d} b \mathrm{d} a + \int_{a=0}^{a=x} \int_{b=0}^{b=a} f^- (b; \psi) \mathrm{d} b \mathrm{d} a \\
\nabla_x F(x;\psi) = \int_{b=x}^{b=\infty} f^+(b; \psi) \mathrm{d}b + \int_{b=0}^{b=x} f^-(b; \psi) \mathrm{d}b \\
        \nabla_x^2 F(x;\psi) = -f^+(x; \psi) + f^-(b; \psi) < 0.
$$
For more details, please refers to [this slides](https://j-zin.github.io/files/shape_constraint_NN_slides.pdf). We demonstrate a toy experiment here of the regression task to show its effectiveness on shape constraints. The experimental setting follows that in UMNN and the code logit is based on [this repo](https://github.com/AWehenkel/UMNN).

| <img src="asset/monotone.jpg" width="100%">                  | <img src="asset/inc_concave.jpg" width="100%">               | <img src="asset/concave.jpg" width="100%">                   |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| $f(\boldsymbol{x}) = 0.001({x}_1^3 + {x}_1) + {x}_2 + \sin({x}_3)$ | $f(\boldsymbol{x}) = 0.001(-e^{-x_1} + {x}_1) + {x}_2 + \sin({x}_3)$ | $f(\boldsymbol{x}) = 0.001(-{x}_1^2 + {x}_1) + {x}_2 + \sin({x}_3)$ |

